#pragma once 
#include <typeinfo>
#include <enoki/stl.h>
#include <enoki/fwd.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/records.h>
#include <mitsuba/core/warp.h>
#include <random>
#include "photonmap.h"
#include "bre.h"

#define M_PI 3.14159265358979323846

template <class T> constexpr std::string_view type_name() {
    using namespace std;
#ifdef __clang__
    string_view p = __PRETTY_FUNCTION__;
    return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
    string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
    return string_view(p.data() + 36, p.size() - 36 - 1);
#else
    return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
    string_view p = __FUNCSIG__;
    return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef BeamRadianceEstimator<Float, Spectrum> BeamRadianceEstimator;
    typedef typename PhotonMap::PhotonData PhotonData;

    PhotonMapper(const Properties &props) : Base(props) {


        m_directSamples    = props.int_("directSamples", 16);
        m_glossySamples    = props.int_("glossySamples", 32);
        m_rrStartDepth     = props.int_("rrStartDepth", 0);
        m_maxDepth         = props.int_("maxDepth", 128);
        m_maxSpecularDepth = props.int_("maxSpecularDepth", 4);
        m_granularity      = props.int_("granularity", 0);
        m_globalPhotons    = props.int_("globalPhotons", 50000);
        m_causticPhotons   = props.int_("causticPhotons", 250000);
        m_volumePhotons    = props.int_("volumePhotons", 250000);
        m_globalLookupRadiusRelative =
            props.float_("globalLookupRadiusRelative", 0.05f);
        m_causticLookupRadiusRelative =
            props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize    = props.int_("globalLookupSize", 50);
        m_causticLookupSize   = props.int_("causticLookupSize", 50);
        m_volumeLookupSize    = props.int_("volumeLookupSize", 50);
        m_gatherLocally       = props.bool_("gatherLocally", true);
        m_autoCancelGathering = props.bool_("autoCancelGathering", true);       

        m_globalPhotonMap  = new PhotonMap(m_globalPhotons);
        m_causticPhotonMap = new PhotonMap(m_causticPhotons);
        m_volumePhotonMap  = new PhotonMap(m_volumePhotons);
        m_bre              = new BeamRadianceEstimator();
    }

    void preprocess(Scene* scene, Sensor* sensor) override {
        static bool m_isPreProcessed = false;
        Log(LogLevel::Info, "Pre Processing...");     
 
        if (!m_isPreProcessed) {
            m_isPreProcessed = true;
            preProcess(scene, sensor->sampler()->clone());
        }
        Log(LogLevel::Info, "Pre Processing done.");     
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static Float m_globalLookupRadius = -1, m_causticLookupRadius = -1;
        if (m_globalLookupRadius == -1) {
            Float sceneRadius =
                norm(scene->bbox().center() - scene->bbox().max);
            m_globalLookupRadius  = m_globalLookupRadiusRelative * sceneRadius;
            m_causticLookupRadius = m_causticLookupRadiusRelative * sceneRadius;
            std::string lookupString = "- Global Lookup Radius: " +
                                       std::to_string(m_globalLookupRadius);
            Log(LogLevel::Info, lookupString.c_str());
            lookupString = "- Scene Radius: " + std::to_string(sceneRadius);
            Log(LogLevel::Info, lookupString.c_str());
        }

        Ray3f r(ray);
        Spectrum radiance(0.0f), throughput(1.0f);
        SurfaceInteraction3f si;

        int depth    = 1;
        int maxDepth = m_maxDepth == -1 ? INT_MAX : (m_maxDepth);

        for (int bounce = 0; bounce < maxDepth; bounce++) {
            si = scene->ray_intersect(r);
            if (!si.is_valid()) {
                active &= bounce > 0;
                break;
            }           

            const BSDF *bsdf = si.bsdf();

            bool isDiracDelta = has_flag(bsdf->flags(), BSDFFlags::Delta);

            BSDFContext bCtx(TransportMode::Radiance);
            auto [bs, bsdfValue] = bsdf->sample(bCtx, si, sampler->next_1d(), sampler->next_2d());
            bsdfValue = si.to_world_mueller(bsdfValue, -bs.wo, si.wi);

            bool rayIsDiracDelta = !has_flag(bs.sampled_type, BSDFFlags::Diffuse);

            if (si.shape->is_emitter()) // && not in medium
            {
                radiance += si.shape->emitter()->eval(si) * throughput;
            }

            if (isDiracDelta) {
                if (bs.pdf < 0)
                    break;
                throughput *= bsdfValue;
            } else {
                // radiance += estimateCausticRadiance(interaction) * throughput;
                radiance += m_globalPhotonMap->estimateRadiance(si, sampler, m_globalLookupRadius, m_globalLookupSize) * throughput;
                break;
            }

            if (absorb(ray, throughput, sampler, bounce)) {
                break;
            }
        }

        return { radiance, active };
    }

    bool absorb(const Ray3f& ray, Spectrum& throughput, Sampler *sampler, int depth) const { 
        float survive = enoki::hmax(throughput);
        if (survive == 0.0f) 
            return true;
        if (depth >= m_rrStartDepth) {
            survive = enoki::min(0.95f, survive);
            if (survive <= sampler->next_1d())
                return true;
            throughput /= survive;
        }
        return false;
    }

    Spectrum Li(RayDifferential3f &ray, const SurfaceInteraction3f &si,
                const Scene *scene, Sampler *sampler, int &depth) const {
        static Float m_globalLookupRadius = -1, m_causticLookupRadius = -1;
        if (m_globalLookupRadius == -1) {
            Float sceneRadius =
                norm(scene->bbox().center() - scene->bbox().max);
            m_globalLookupRadius  = m_globalLookupRadiusRelative * sceneRadius;
            m_causticLookupRadius = m_causticLookupRadiusRelative * sceneRadius;
            std::string lookupString = "- Global Lookup Radius: " +
                                       std::to_string(m_globalLookupRadius);
            Log(LogLevel::Info, lookupString.c_str());
            lookupString = "- Scene Radius: " + std::to_string(sceneRadius);
            Log(LogLevel::Info, lookupString.c_str());
        }

        Spectrum LiSurf(0.0f), LiMedium(0.0f), transmittance(1.0f);

        if (!si.is_valid()) {
            if (!m_hide_emitters && scene->environment() != nullptr)
                LiSurf = scene->environment()->eval(si);
            return LiSurf * transmittance + LiMedium;
        }

        UInt32 mediumChannel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            mediumChannel =
                (UInt32) min(sampler->next_1d() * n_channels, n_channels - 1);
        }        

        MediumPtr medium    = si.target_medium(ray.d);
        Mask active_medium = neq(medium, nullptr);
        if (any_or<true>(active_medium)) {

            Ray mediumRaySegment(ray, 0, si.t);
            Mask is_spectral       = active_medium;
            MediumInteraction3f mi = medium->sample_interaction(
                ray, sampler->next_1d(), mediumChannel, active_medium);
            auto [tr, pdf] = medium->eval_tr_and_pdf(mi, si, is_spectral);
            transmittance         = tr;
            mediumRaySegment.mint = ray.mint;
            if ((depth < m_maxDepth || m_maxDepth < 0) && m_volumePhotonMap->size() > 0)
                LiMedium = m_bre->query(mediumRaySegment, medium, si, sampler, mediumChannel, active_medium, m_maxDepth, true);
        }

        /* Possibly include emitted radiance if requested */
        if (si.shape->is_emitter() && !m_hide_emitters)
            LiSurf += si.shape->emitter()->eval(si);

        const BSDF *bsdf = si.bsdf();

        if (depth >= m_maxDepth && m_maxDepth > 0)
            return LiSurf * transmittance + LiMedium;

        /* Irradiance cache query -> treat as diffuse */
        bool isDiffuse   = has_flag(bsdf->flags(), BSDFFlags::DiffuseReflection);
        bool hasSpecular = has_flag(bsdf->flags(), BSDFFlags::Delta);
        /* Exhaustively recurse into all specular lobes? */
        bool exhaustiveSpecular = depth < m_maxSpecularDepth;

        if (isDiffuse && (dot(si.sh_frame.n, ray.d) < 0 || has_flag(bsdf->flags(), BSDFFlags::BackSide))) {
            /* 1. Diffuse indirect */
            int maxDepth = m_maxDepth == -1 ? INT_MAX : (m_maxDepth - depth);
            
            BSDFContext ctx1(TransportMode::Radiance);
            LiSurf += m_globalPhotonMap->estimateIrradiance(
                          si.p, si.sh_frame.n, m_globalLookupRadius,
                            maxDepth, m_globalLookupSize) *
                        bsdf->eval(ctx1, si, Vector3f(0, 0, 1));
            
            if (false) {
                BSDFContext ctx2(TransportMode::Radiance);
                LiSurf += m_causticPhotonMap->estimateIrradiance(
                              si.p, si.sh_frame.n, m_causticLookupRadius,
                              maxDepth, m_causticLookupSize) *
                          bsdf->eval(ctx2, si, Vector3f(0, 0, 1));
            }
        }
       
        return LiSurf * transmittance + LiMedium;
    }


    void preProcess(const Scene *scene, Sampler *sampler) {
        Log(LogLevel::Info, "Pre Processing Photon Map...");
        printEmitters(scene);

        sampler->seed(0);

        MediumInteraction3f mi;
        SurfaceInteraction3f si;
        Interaction3f its;
        ref<Sensor> sensor = scene->sensors()[0];
        Float time = sensor->shutter_open() + 0.5f * sensor->shutter_open_time();
        int numShot = 0;



        while (m_globalPhotonMap->size() < m_globalPhotons) {
            sampler->advance();

            std::string debugStr = "- Photon Num: " + std::to_string(++numShot);
            Log(LogLevel::Info, debugStr.c_str());
            MediumPtr medium;
            // Sample random emitter
            auto tuple = sample_emitter_direction(scene, its, sampler->next_2d(), true);
            EmitterPtr emitter = std::get<2>(tuple);
            // Sample random ray from emitter
            auto rayColorPair = emitter->sample_ray(time, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            RayDifferential3f ray(rayColorPair.first);

            Spectrum power(1.0f / m_globalPhotons), bsdf_absIdotN;

            Mask active = true;

            UInt32 channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel = (UInt32) min(sampler->next_1d(active) * n_channels,
                                       n_channels - 1);
            }

            bool isTransmittedPhoton = false;
            bool isDiracDelta = false;

            for (int depth = 0; depth < m_maxDepth; ++depth) {
                si = scene->ray_intersect(ray);
                if (!si.is_valid())
                    break;

                const BSDF *bsdf = si.bsdf();

                if (has_flag(bsdf->flags(), BSDFFlags::Diffuse)) {
                    if (!isDiracDelta) {
                        m_globalPhotonMap->insert(
                            si.p, PhotonData(si.n, -ray.d, power, depth));
                    } else {
                        m_causticPhotonMap->insert(
                            si.p, PhotonData(si.n, -ray.d, power, depth));
                    }
                }

                BSDFContext bCtx(TransportMode::Radiance);
                auto [bs, bsdfValue] = bsdf->sample(bCtx, si, sampler->next_1d(), sampler->next_2d());
                bsdfValue = si.to_world_mueller(bsdfValue, -bs.wo, si.wi);
                if (bs.pdf < 0)
                    break;

                Vector wi = -ray.d, wo = si.to_world(bs.wo);
                Float wiDotGeoN = dot(si.n, wi),
                      woDotGeoN = dot(si.n, wo);
                if (wiDotGeoN * Frame3f::cos_theta(si.wi) <= 0 ||
                    woDotGeoN * Frame3f::cos_theta(bs.wo) <= 0)
                    break;

                bsdf_absIdotN = bsdfValue;

                isDiracDelta = has_flag(bs.sampled_type, BSDFFlags::Delta);

                if (depth >= m_rrStartDepth) {
                    float survive = enoki::min(enoki::hmax(bsdf_absIdotN), (float) 0.95f);
                    if (survive == 0.0 || sampler->next_1d() <= survive)
                        break;
                    power *= bsdf_absIdotN / survive;
                }

                ray = si.spawn_ray(si.to_world(bs.wo));
            }

        }

        
        Log(LogLevel::Info, "Building Global Photon Map...");
        m_globalPhotonMap->setScaleFactor(1.0f / m_globalPhotonMap->size());
        m_globalPhotonMap->build();
        Log(LogLevel::Info, "Finished Global Photon Map...");

        if (m_causticPhotonMap->size() > 0) {
            Log(LogLevel::Info, "Building Caustic Photon Map...");
            m_causticPhotonMap->setScaleFactor(1.0f / m_causticPhotonMap->size());
            m_causticPhotonMap->build();
            Log(LogLevel::Info, "Finished Caustic Photon Map...");
        }

        if (m_volumePhotonMap->size() > 0) {
            Log(LogLevel::Info, "Building Volume Photon Map...");
            m_volumePhotonMap->setScaleFactor(1.0f / m_volumePhotonMap->size());
            m_volumePhotonMap->build();
            m_bre->build(m_volumePhotonMap, m_volumeLookupSize);
            Log(LogLevel::Info, "Finished Volume Photon Map...");
        }
    }

    void
    printEmitters(const mitsuba::PhotonMapper<Float, Spectrum>::Scene *scene) {
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string photonString =
            "- Emitter Count: " + std::to_string(emitters.size());
        Log(LogLevel::Info, photonString.c_str());

        for (int i = 0; i < emitters.size(); i++) {
            std::string emitterType =
                "- E" + std::to_string(i) + " = " + typeid(&emitters[i]).name();
            Log(LogLevel::Info, emitterType.c_str());
        }
    }

    std::tuple<DirectionSample3f, Spectrum, EmitterPtr>
    sample_emitter_direction(const Scene *scene, const Interaction3f &ref,
                             const Point2f &sample_,
                             Mask active) const {

        Point2f sample(sample_);
        DirectionSample3f ds;
        Spectrum spec;
        EmitterPtr emitter = nullptr;

        if (likely(!scene->emitters().empty())) {
            if (scene->emitters().size() == 1) {
                std::tie(ds, spec) =
                    scene->emitters()[0]->sample_direction(ref, sample, active);
                emitter = scene->emitters()[0];
            } else {
                ScalarFloat emitter_pdf = 1.f / scene->emitters().size();
                UInt32 index = min(UInt32(sample.x() * (ScalarFloat) scene->emitters().size()), (uint32_t) scene->emitters().size() - 1);
                sample.x() = (sample.x() - index * emitter_pdf) * scene->emitters().size();
                emitter = scene->emitters()[index];
                std::tie(ds, spec) = emitter->sample_direction(ref, sample, active);
                ds.pdf *= emitter_pdf;
                //spec *= rcp(emitter_pdf);
            }

            active &= neq(ds.pdf, 0.f);
        } else {
            ds   = zero<DirectionSample3f>();
            spec = 0.f;
        }

        return std::make_tuple(ds, spec, emitter);
    }

    void handleMediumInteraction(int _depth, int nullInteractions, bool delta,
                                 const MediumInteraction3f &mi,
                                 const Medium *medium, const Vector3f &wi,
                                 const Spectrum &weight) const {
        int depth = _depth - nullInteractions;
        if (depth < m_minDepth) {
            return;
        }
        m_volumePhotonMap->insert(mi.p, PhotonData(Normal3f(0.0f, 0.0f, 0.0f), -wi, weight, depth));
    }

    MTS_DECLARE_CLASS()
private:
    PhotonMap *m_globalPhotonMap;
    PhotonMap *m_causticPhotonMap;
    PhotonMap *m_volumePhotonMap;
    BeamRadianceEstimator *m_bre;

    int m_directSamples, m_glossySamples, m_rrStartDepth, m_maxDepth,
        m_maxSpecularDepth, m_granularity;
    int m_minDepth = 0;
    int m_globalPhotons, m_causticPhotons, m_volumePhotons;
    float m_globalLookupRadiusRelative;
    float m_causticLookupRadiusRelative;
    float m_invEmitterSamples, m_invGlossySamples;
    int m_globalLookupSize, m_causticLookupSize, m_volumeLookupSize;
    /* Should photon gathering steps exclusively run on the local machine? */
    bool m_gatherLocally;
    /* Indicates if the gathering steps should be canceled if not enough photons
     * are generated. */
    bool m_autoCancelGathering;
};

MTS_IMPLEMENT_CLASS_VARIANT(PhotonMapper, SamplingIntegrator);
MTS_EXPORT_PLUGIN(PhotonMapper, "Photon Mapping integrator");
NAMESPACE_END(mitsuba)
