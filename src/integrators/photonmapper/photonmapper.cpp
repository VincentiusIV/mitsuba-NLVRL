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

const int k          = 3;
const int numPhotons = 100000;

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
        m_globalPhotonMap    = new PhotonMap(numPhotons);
        m_causticPhotonMap = new PhotonMap(numPhotons);
        m_volumePhotonMap    = new PhotonMap(numPhotons);       
        m_bre = new BeamRadianceEstimator();

        m_directSamples    = props.int_("directSamples", 16);
        m_glossySamples    = props.int_("glossySamples", 32);
        m_rrStartDepth     = props.int_("rrStartDepth", 5);
        m_maxDepth         = props.int_("maxDepth", 128);
        m_maxSpecularDepth = props.int_("maxSpecularDepth", 4);
        m_granularity      = props.int_("granularity", 0);
        m_globalPhotons    = props.int_("globalPhotons", 250000);
        m_causticPhotons   = props.int_("causticPhotons", 250000);
        m_volumePhotons    = props.int_("volumePhotons", 250000);
        m_globalLookupRadiusRelative =
            props.float_("globalLookupRadiusRelative", 0.1f);
        m_causticLookupRadiusRelative =
            props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize    = props.int_("globalLookupSize", 120);
        m_causticLookupSize   = props.int_("causticLookupSize", 120);
        m_volumeLookupSize    = props.int_("volumeLookupSize", 120);
        m_gatherLocally       = props.bool_("gatherLocally", true);
        m_autoCancelGathering = props.bool_("autoCancelGathering", true);       
    }

    void preprocess(Scene* scene, Sensor* sensor) override {
        Log(LogLevel::Info, "Pre Processing...");     
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        Sampler *sampler = sensor->sampler();
        sampler->seed(0);

        std::string numPhotonsStr =
            "- Photon Count: " + std::to_string(numPhotons);
        Log(LogLevel::Info, numPhotonsStr.c_str());
        // 1. For each light source in the scene we create a set of photons
        //    and divide the overall power of the light source amongst them.
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string photonString =
            "- Emitter Count: " + std::to_string(emitters.size());
        Log(LogLevel::Info, photonString.c_str());

        for (int i = 0; i < emitters.size(); i++) {
            std::string emitterType =
                "- E" + std::to_string(i) + " = " + typeid(&emitters[i]).name();
            Log(LogLevel::Info, emitterType.c_str());
        }

        MediumInteraction3f mi;
        SurfaceInteraction3f si;
        Interaction3f its;
        Float time = sensor->shutter_open() + 0.5f * sensor->shutter_open_time();

        int numShot = 0;

        for (int index = 0; index < numPhotons; index++) {
            sampler->advance();

            EmitterPtr emitter;
            MediumPtr medium;
            Spectrum power(1.0f);

            // Sample random emitter
            auto tuple = sample_emitter_direction(scene, its, sampler->next_2d(), false, true);
            emitter = std::get<2>(tuple);

            auto rayColorPair = emitter->sample_ray(time, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            RayDifferential3f ray(rayColorPair.first);

            int depth = 1, nullInteractions = 0;
            bool delta               = false;
            bool lastNullInteraction = false;

            Spectrum throughput(1.0f);
            Mask active             = true;
            Mask needs_intersection = true;

            UInt32 channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel = (UInt32) min(sampler->next_1d(active) * n_channels,
                                       n_channels - 1);
            }
            ++numShot;

            while (throughput != Spectrum(0.0f) && (depth <= m_maxDepth || m_maxDepth < 0)) {

                si = scene->ray_intersect(ray);

                if (!si.is_valid()) {
                    break;
                }

                const BSDF *bsdf = si.bsdf();

                /* Forward the surface scattering event to the attached
                    * * handler */
                handleSurfaceInteraction(depth, nullInteractions, delta, si,
                                            medium, throughput * power);
                BSDFContext bCtx(TransportMode::Radiance);
                auto [bs, bsdfWeight] = bsdf->sample(
                    bCtx, si, sampler->next_1d(), sampler->next_2d());
                bsdfWeight = si.to_world_mueller(bsdfWeight, -bs.wo, si.wi);
                if (bsdfWeight == Spectrum(0.0f)) {
                    Log(LogLevel::Info, "BSDF Weight is zero :(");
                    break;
                }

                /* Prevent light leaks due to the use of shading normals
                    * * -- [Veach, p. 158] */
                Vector3f wi = -ray.d, wo = si.to_world(bs.wo);
                Float wiDotGeoN = dot(si.n, wi), woDotGeoN = dot(si.n, wo);
                if (wiDotGeoN * Frame3f::cos_theta(si.wi) <= 0 ||
                    woDotGeoN * Frame3f::cos_theta(bs.wo) <= 0) {
                    break;
                }

                /* Keep track of the weight, medium and relative
                    refractive index along the path */
                throughput *= bsdfWeight;
                if (si.is_medium_transition())
                    medium = si.target_medium(woDotGeoN);

                if (bs.sampled_type & BSDFFlags::Null) {
                    ++nullInteractions;
                    lastNullInteraction = true;
                } else {
                    delta = bs.sampled_type & BSDFFlags::Delta;
                    lastNullInteraction = false;
                }

                /* Adjoint BSDF for shading normals -- [Veach, p. 155]
                    */
                if (true) {
                    throughput *=
                        std::abs((Frame3f::cos_theta(-bs.wo) * woDotGeoN) /
                                    (Frame3f::cos_theta(bs.wo) * wiDotGeoN));
                }

                handleSurfaceInteractionScattering(bs, si, ray);
                

                if (depth++ >= m_rrStartDepth) {
                    Float q =
                        enoki::min(enoki::hmax(throughput), (Float) 0.95f);
                    if (sampler->next_1d() >= q)
                        break;
                    throughput /= q;
                }
            }
        }

        float scale          = 1.0 / m_globalPhotonMap->size();
        std::string debugStr = "Global Photon scale: " + std::to_string(scale);
        Log(LogLevel::Info, debugStr.c_str());
        debugStr = "Num shot: " + std::to_string(numShot);
        Log(LogLevel::Info, debugStr.c_str());
        m_globalPhotonMap->setScaleFactor(1.0 / numShot);
        m_globalPhotonMap->build();

        if (m_causticPhotonMap->size() > 0) {
            m_causticPhotonMap->setScaleFactor(1.0f / numShot);
            m_causticPhotonMap->build();
        }

        if (m_volumePhotonMap->size() > 0) {
            m_volumePhotonMap->setScaleFactor(1.0f / numShot);
            m_volumePhotonMap->build();
            m_bre->build(m_volumePhotonMap, m_volumeLookupSize);
        }
        Log(LogLevel::Info, "Pre Processing done.");     
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &_ray,
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


        RayDifferential3f ray(_ray);
        Spectrum radiance(0.0f), throughput(1.0f);
        float eta(1.0f);

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray = si.is_valid();

        for (int depth = 1;; ++depth) {

            if (si.is_valid())
            {   
                //if (si.shape->is_emitter() && !m_hide_emitters)
                //    radiance += si.shape->emitter()->eval(si, active) * throughput;
            }

            active &= si.is_valid();

            // Russian roulette
            if (depth > m_rrStartDepth) {
                Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                active &= sampler->next_1d(active) < q;
                throughput *= rcp(q);
            }

            if ((uint32_t) depth >= (uint32_t) m_maxDepth ||
                ((!is_cuda_array_v<Float> || m_maxDepth < 0) && none(active)))
                break;

            BSDFContext bCtx;
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);
            
            // Photon Map Sampling
            if (likely(any_or<true>(active_e))) {
                radiance += m_causticPhotonMap->estimateRadiance(si, m_causticLookupRadius, m_causticLookupSize) * throughput;
                radiance += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize) * throughput;
            }

            auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active), sampler->next_2d(active), active);            
            bsdfVal = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

            throughput = throughput * bsdfVal;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            eta *= bs.eta;

            ray = si.spawn_ray(si.to_world(bs.wo));
            si = scene->ray_intersect(ray, active);    
        }
        
        return { radiance, valid_ray };
    }


    MTS_INLINE Float index_spectrum(const UnpolarizedSpectrum &spec,
                         const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m, eq(idx, 1u)) = spec[1];
            masked(m, eq(idx, 2u)) = spec[2];
        } else {
            ENOKI_MARK_USED(idx);
        }
        return m;
    }

    std::tuple<DirectionSample3f, Spectrum, EmitterPtr>
    sample_emitter_direction(const Scene *scene, const Interaction3f &ref,
                             const Point2f &sample_, bool test_visibility,
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
                spec *= rcp(emitter_pdf);
            }

            active &= neq(ds.pdf, 0.f);

            if (test_visibility && any_or<true>(active)) {
                Ray3f ray(ref.p, ds.d, math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))), ds.dist * (1.f - math::ShadowEpsilon<Float>), ref.time, ref.wavelengths);
                spec[scene->ray_test(ray, active)] = 0.f;
            }
        } else {
            ds   = zero<DirectionSample3f>();
            spec = 0.f;
        }

        return std::make_tuple(ds, spec, emitter);
    }

    void handleSurfaceInteraction(int _depth, int nullInteractions, bool delta,
                                  const SurfaceInteraction3f &si,
                                  const Medium *medium,
                                  const Spectrum &weight) const {
        BSDFContext ctx;
        BSDFPtr bsdf       = si.bsdf();
        int depth          = _depth - nullInteractions;
        uint32_t bsdfFlags = bsdf->flags();
        if (!(bsdfFlags & BSDFFlags::DiffuseReflection) &&
            !(bsdfFlags & BSDFFlags::GlossyReflection))
            return;
        if (depth < m_minDepth)
            return;
        if (!delta)
            m_globalPhotonMap->insert(si.p, PhotonData(si.n, -si.to_world(si.wi), weight, depth));
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

    void handleSurfaceInteractionScattering(const BSDFSample3f &bRec,
                                            const SurfaceInteraction3f &si,
                                            RayDifferential3f &ray) const { 
        ray.o    = si.p;
        ray.d    = si.to_world(bRec.wo);
        ray.mint = math::RayEpsilon<Float>;
    }

    MTS_DECLARE_CLASS()
private:
    PhotonMap *m_globalPhotonMap;
    PhotonMap *m_causticPhotonMap;
    PhotonMap *m_volumePhotonMap;
    BeamRadianceEstimator *m_bre;

    int m_directSamples, m_glossySamples, m_rrStartDepth, m_maxDepth,
        m_maxSpecularDepth, m_granularity;
    int m_minDepth = 1;
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
