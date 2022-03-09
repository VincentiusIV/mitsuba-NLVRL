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
#include <random>
#include "kdtree.h"

#define INV_PI 0.31830988618379067154

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

const int k = 3;

template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    struct PointNode {
        float point[k];
        short flag;
        PointNode *left;
        PointNode *right;        
    };

    struct Photon : PointNode {
        Spectrum spectrum;
        Normal3f normal;
        Vector3f direction;
        int depth;

        Photon(const Point3f &position, const Normal3f &normal,
               const Vector3f &direction, const Spectrum &spectrum,
               const int &depth)
            : PointNode() {
            point[0]        = position[0];
            point[1]        = position[1];
            point[2]        = position[2];
            this->spectrum  = spectrum;
            this->normal    = normal;
            this->direction = direction;
            this->depth     = depth;
        }
    };

    class PhotonMap {
    public:
        PhotonMap() { 
            Log(LogLevel::Info, "Constructing PhotonMap...");
            m_scale = 0.001f;
        }

        inline void insert(const Photon &photon) { 
            photons.push_back(photon);
            std::string photonString =
                "Inserting photon... Count: " + std::to_string(photons.size());
            Log(LogLevel::Info, photonString.c_str());       
            std::string numPhotonsStr = "px: " + std::to_string(photon.point[0]) +
                "py: " + std::to_string(photon.point[1]) +
                "pz: " + std::to_string(photon.point[2]);
            Log(LogLevel::Info, numPhotonsStr.c_str());  

            std::string colorStr =
                "r: " + std::to_string(photon.spectrum[0]) +
                "g: " + std::to_string(photon.spectrum[1]) +
                "b: " + std::to_string(photon.spectrum[2]);
            Log(LogLevel::Info, colorStr.c_str());  

            std::string normalStr = "nx: " + std::to_string(photon.normal[0]) +
                "ny: " + std::to_string(photon.normal[1]) +
                "nz: " + std::to_string(photon.normal[2]);
            Log(LogLevel::Info, normalStr.c_str());  
        }       

        Spectrum estimateRadiance(const SurfaceInteraction3f& si,
                                  float searchRadius, size_t maxPhotons) const {
            float squaredRadius = searchRadius * searchRadius;
            size_t resultCount = photons.size(); // nnSearch(p, squaredRadius,
                                                 // maxPhotons, results);
            float invSquaredRadius = 1.0f / squaredRadius;

            /* Sum over all contributions */
            Spectrum result(0.0f);
            const BSDF *bsdf = si.bsdf();
            for (size_t i = 0; i < resultCount; i++) {
                Photon photon = photons[i];
                Point3f photonP(photon.point[0], photon.point[1], photon.point[2]);
                Vector3f between  = photonP - si.p;
                float distSquared = between[0] * between[0] +
                                    between[1] * between[1] +
                                    between[2] * between[2];
                float sqrTerm = 1.0f - distSquared * invSquaredRadius;
                if (distSquared > squaredRadius)
                    continue;

                Vector3f wi = si.to_local(-photon.direction);

                BSDFContext bRec(TransportMode::Importance);
                result += photon.spectrum * bsdf->eval(bRec, si, si.wi) * (sqrTerm * sqrTerm);
            }

            /* Based on the assumption that the surface is locally flat,
               the estimate is divided by the area of a disc corresponding to
               the projected spherical search region */
            return result * (m_scale * 3.0f * INV_PI * invSquaredRadius);
        }

        Spectrum estimateIrradiance(const Point3f &p, const Normal3f &n,
                                    float searchRadius, int maxDepth,
                                    size_t maxPhotons) const {
            //SearchResult *results = static_cast<SearchResult *>(alloca((maxPhotons + 1) * sizeof(SearchResult)));
            float squaredRadius = searchRadius * searchRadius;
            size_t resultCount  = min(maxPhotons, photons.size()); // nnSearch(p, squaredRadius,
                                                 // maxPhotons, results);
            float invSquaredRadius = 1.0f / squaredRadius;
            /* Sum over all contributions */
            Spectrum result(0.0f);
            for (size_t i = 0; i < resultCount; i++) {
                //const SearchResult &searchResult = results[i];
                Photon photon = photons[i];
                if (photon.depth > maxDepth)
                    continue;
                Point3f photonP(photon.point[0], photon.point[1], photon.point[2]);

                Vector3f between   = photonP - p;
                float distSquared =
                    between[0] * between[0] + 
                    between[1] * between[1] +
                    between[2] * between[2];
                float sqrTerm = 1.0f - distSquared * invSquaredRadius;

                if (distSquared > squaredRadius)
                    continue;

                Vector3f wi = -photon.direction;
                float wiDotGeoN = dot(photon.normal, wi);
                float wiDotShN  = dot(n, wi);

                /* Only use photons from the top side of the surface */
                if (dot(wi, n) > 0 && dot(photon.normal, n) > 1e-1f && wiDotGeoN > 1e-2f) {
                    /* Account for non-symmetry due to shading normals */
                    Spectrum power = photon.spectrum* std::abs(wiDotShN / wiDotGeoN);

                    /* Weight the samples using Simpson's kernel */
                    float sqrTerm = 1.0f - distSquared * invSquaredRadius;

                    result += power * (sqrTerm * sqrTerm);
                }

            }
            /* Based on the assumption that the surface is locally flat,
               the estimate is divided by the area of a disc corresponding to
               the projected spherical search region */
            return result * (m_scale * 3.0f * INV_PI * invSquaredRadius);
        }



    private:
        std::vector<Photon> photons;
        float m_scale;
    };

    PhotonMapper(const Properties &props) : Base(props) {
        m_globalPhotonMap  = new PhotonMap();
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
            props.float_("globalLookupRadiusRelative", 0.05f);
        m_causticLookupRadiusRelative =
            props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize    = props.int_("globalLookupSize", 120);
        m_causticLookupSize   = props.int_("causticLookupSize", 120);
        m_volumeLookupSize    = props.int_("volumeLookupSize", 120);
        m_gatherLocally       = props.bool_("gatherLocally", true);
        m_autoCancelGathering = props.bool_("autoCancelGathering", true);
       
    }

    void preprocess(Scene* scene, Sensor* sensor) const override {
        static bool m_isPreProcessed = false;
        Log(LogLevel::Info, "Pre Processing...");     
 
        if (!m_isPreProcessed) {
            m_isPreProcessed = true;
            preProcess(scene, sensor->sampler()->clone());
        }
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static Float m_globalLookupRadius = -1, m_causticLookupRadius = -1;
        if (m_globalLookupRadius == -1) {
            Float sceneRadius = norm(scene->bbox().center() - scene->bbox().max);
            m_globalLookupRadius = m_globalLookupRadiusRelative * sceneRadius;
            m_causticLookupRadius = m_causticLookupRadiusRelative * sceneRadius;
            std::string lookupString = "- Global Lookup Radius: " + std::to_string(m_globalLookupRadius);
            Log(LogLevel::Info, lookupString.c_str());   
            lookupString = "- Scene Radius: " + std::to_string(sceneRadius);
            Log(LogLevel::Info, lookupString.c_str());   
        }

        SurfaceInteraction3f si = scene->ray_intersect(ray);

        Spectrum LiSurf(0.0f), LiMedium(0.0f), transmittance(1.0f);
        int maxDepth = m_maxDepth == -1 ? INT_MAX : (m_maxDepth); // - rRec.depth)

        if (si.is_valid()) {

            const BSDF *bsdf = si.bsdf();
            BSDFContext bCtx(TransportMode::Radiance);
            auto [bs, bsdf_val] = bsdf->sample(bCtx, si, sampler->next_1d(active), sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            LiSurf += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize);
            LiSurf += m_globalPhotonMap->estimateIrradiance(si.p, si.sh_frame.n, m_globalLookupRadius, maxDepth, m_globalLookupSize) * bsdf_val;

            
        }

        return { LiSurf, si.is_valid() };
    }

    void preProcess(const Scene *scene, Sampler *sampler) const {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        sampler->seed(0);

        const int numPhotons = 10000;
        std::string numPhotonsStr =
            "- Photon Count: " + std::to_string(numPhotons);
        Log(LogLevel::Info, numPhotonsStr.c_str());    
        // 1. For each light source in the scene we create a set of photons
        //    and divide the overall power of the light source amongst them.       
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string photonString = "- Emitter Count: " + std::to_string(emitters.size());
        Log(LogLevel::Info, photonString.c_str());    

        for (int i = 0; i < emitters.size(); i++) {
            std::string emitterType = "- E" + std::to_string(i) + " = " + typeid(&emitters[i]).name();
            Log(LogLevel::Info, emitterType.c_str());    
        }

        MediumInteraction3f mRec;
        SurfaceInteraction3f si;
        Interaction3f its;
        ref<Sensor> sensor = scene->sensors()[0];
        Float time = sensor->shutter_open() + 0.5f * sensor->shutter_open_time();

        for (int index = 0; index < numPhotons; index++) {
            std::string debugStr =  "- Photon Num: " + std::to_string(index);
            Log(LogLevel::Info, debugStr.c_str());   
            
            //sampler->seed(index);

            EmitterPtr emitter;
            const Medium *medium;
            Spectrum power(1.0f);

            Log(LogLevel::Info, "Sampling emitter direction");   

            auto tuple = sample_emitter_direction(
                scene, its, sampler->next_2d(), false, true);
            //power   = std::get<1>(tuple);
            emitter = std::get<2>(tuple);
            medium  = emitter->medium();

            Log(LogLevel::Info, "Sampling ray from emitter");   

            auto rayColorPair = emitter->sample_ray(time, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            RayDifferential3f ray(rayColorPair.first);

            int depth = 1, nullInteractions = 0;
            bool delta               = false;
            bool lastNullInteraction = false;

            Log(LogLevel::Info, "Tracing ray");   
            Spectrum throughput(1.0f);
            while (throughput != Spectrum(0.0f) &&
                   (depth <= m_maxDepth || m_maxDepth < 0)) {
                si = scene->ray_intersect(ray);
                if (false) { // medium && medium->sampleDistance(Ray(ray, 0,
                             // its.t), mRec, m_sampler)
                    /*throughput *= mRec.sigma_s * mRec.transmittance /
                                        mRec.pdfSuccess;
                    handleMediumInteraction(depth, nullInteractions, delta,
                    mRec, medium, -ray.d, throughput * power);
                    PhaseFunctionSamplingRecord pRec(mRec, -ray.d,
                    TransportMode::Importance); throughput *=
                    medium->getPhaseFunction()->sample(pRec, m_sampler);
                    delta = false;
                    lastNullInteraction = false;
                    handleMediumInteractionScattering(mRec, pRec, ray);*/
                } else if (!si.is_valid()) {
                    break;
                } else {
                    /* Sample
                        tau(x, y) (Surface integral). This happens with
                        probability mRec.pdfFailure Account for this and
                        multiply by the proper per-color-channel
                        transmittance.
                    */
                    // if (medium)
                    //    throughput *= mRec.transmittance /
                    //    mRec.pdfFailure;
                    const BSDF *bsdf = si.bsdf();

                    /* Forward the surface scattering event to the attached
                     * * handler */
                    handleSurfaceInteraction(depth, nullInteractions, delta, si,
                                             medium, throughput * power);
                    BSDFContext bCtx(TransportMode::Importance);
                    std::pair<BSDFSample3f, Spectrum> _sample = bsdf->sample(
                        bCtx, si, sampler->next_1d(), sampler->next_2d());
                    BSDFSample3f bsdfSample = _sample.first;
                    Spectrum bsdfWeight     = _sample.second;
                    if (bsdfWeight == Spectrum(0.0f)) {
                        Log(LogLevel::Info, "BSDF Weight is zero :(");                        
                        break;
                    }

                    /* Prevent light leaks due to the use of shading normals
                     * * -- [Veach, p. 158] */
                    Vector3f wi = -ray.d, wo = si.to_world(bsdfSample.wo);
                    Float wiDotGeoN = dot(si.n, wi), woDotGeoN = dot(si.n, wo);
                    if (wiDotGeoN * Frame3f::cos_theta(-bsdfSample.wo) <= 0 ||
                        woDotGeoN * Frame3f::cos_theta(bsdfSample.wo) <= 0)
                        break;

                    /* Keep track of the weight, medium and relative
                        refractive index along the path */
                    throughput *= bsdfWeight;
                    if (si.is_medium_transition())
                        medium = si.target_medium(woDotGeoN);

                    if (bsdfSample.sampled_type & BSDFFlags::Null) {
                        ++nullInteractions;
                        lastNullInteraction = true;
                    } else {
                        delta = bsdfSample.sampled_type & BSDFFlags::Delta;
                        lastNullInteraction = false;
                    }

                    /* Adjoint BSDF for shading normals -- [Veach, p. 155]
                     */
                    if (true) {
                        throughput *= std::abs(
                            (Frame3f::cos_theta(-bsdfSample.wo) * woDotGeoN) /
                            (Frame3f::cos_theta(bsdfSample.wo) * wiDotGeoN));
                    }

                    handleSurfaceInteractionScattering(bsdfSample, si, ray);
                }

                if (depth++ >= m_rrStartDepth) {
                    Float q =
                        enoki::min(enoki::hmax(throughput), (Float) 0.95f);
                    if (sampler->next_1d() >= q)
                        break;
                    throughput /= q;
                }
            }
        }
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
                UInt32 index            = min(
                    UInt32(sample.x() * (ScalarFloat) scene->emitters().size()),
                    (uint32_t) scene->emitters().size() - 1);
                sample.x() = (sample.x() - index * emitter_pdf) *
                             scene->emitters().size();
                emitter = scene->emitters()[index];
                    //gather<EmitterPtr>(scene->emitters().data(), index, active);
                std::tie(ds, spec) =
                    emitter->sample_direction(ref, sample, active);
                ds.pdf *= emitter_pdf;
                spec *= rcp(emitter_pdf);
            }

            active &= neq(ds.pdf, 0.f);

            if (test_visibility && any_or<true>(active)) {
                Ray3f ray(ref.p, ds.d,
                          math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))),
                          ds.dist * (1.f - math::ShadowEpsilon<Float>),
                          ref.time, ref.wavelengths);
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
            m_globalPhotonMap->insert(Photon(si.p, si.n, -si.to_world(si.wi), weight, depth));
    }

    void handleMediumInteraction(int _depth, int nullInteractions, bool delta,
                                 const MediumInteraction3f &mi,
                                 const Medium *medium, const Vector3f &wi,
                                 const Spectrum &weight) const {
        int depth = _depth - nullInteractions;
        if (depth < m_minDepth) {
            return;
        }
        m_volumePhotonMap->insert(
            Photon(mi.p, Normal3f(0.0f, 0.0f, 0.0f), -wi, weight, depth));
    }

    void handleSurfaceInteractionScattering(BSDFSample3f &bRec,
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

    int m_directSamples, m_glossySamples, m_rrStartDepth, m_maxDepth,
        m_maxSpecularDepth, m_granularity;
    int m_minDepth = 1;
    int m_globalPhotons, m_causticPhotons, m_volumePhotons;
    float m_globalLookupRadiusRelative;
    float m_causticLookupRadiusRelative;
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
