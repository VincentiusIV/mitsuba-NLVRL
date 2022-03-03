#pragma once
#include "kdtree.h"
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

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    struct Photon : PointNode {
        Spectrum spectrum;
        Normal3f normal;
        Vector3f direction;

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
        }
    };

    class PhotonMap : PointKDTree<Photon> {
    public:
        PhotonMap() { Log(LogLevel::Info, "Constructing PhotonMap..."); }

        inline void insert(const Photon &photon) {

        }

        Spectrum estimateIrradiance(const Point3f &p, const Normal3f &n,
                                    Float searchRadius, int maxDepth,
                                    size_t maxPhotons) const {            
            SearchResult *results = static_cast<SearchResult *>(alloca((maxPhotons + 1) * sizeof(SearchResult)));
            Float squaredRadius = searchRadius * searchRadius;
            size_t resultCount =  nnSearch(p, squaredRadius, maxPhotons, results);
            Float invSquaredRadius = 1.0f / squaredRadius;
            /* Sum over all contributions */
            Spectrum result(0.0f);
            for (size_t i = 0; i < resultCount; i++) {
                const SearchResult &searchResult = results[i];
                const Photon &photon             = m_kdtree[searchResult.index];
                if (photon.getDepth() > maxDepth)
                    continue;

                Vector wi           = -photon.getDirection();
                Vector photonNormal = photon.getNormal();
                Float wiDotGeoN = dot(photonNormal, wi), wiDotShN = dot(n, wi);

                /* Only use photons from the top side of the surface */
                if (dot(wi, n) > 0 && dot(photonNormal, n) > 1e-1f &&
                    wiDotGeoN > 1e-2f) {
                    /* Account for non-symmetry due to shading normals */
                    Spectrum power =
                        photon.getPower() * std::abs(wiDotShN / wiDotGeoN);

                    /* Weight the samples using Simpson's kernel */
                    Float sqrTerm =
                        1.0f - searchResult.distSquared * invSquaredRadius;

                    result += power * (sqrTerm * sqrTerm);
                }
            }
            /* Based on the assumption that the surface is locally flat,
               the estimate is divided by the area of a disc corresponding to
               the projected spherical search region */
            return result * (/*m_scale * */ 3 * math::InvPi<Float> * invSquaredRadius);
        }

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

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static bool m_isPreProcessed = false;
        if (!m_isPreProcessed) {
            m_isPreProcessed = true;
            preProcess(scene, sampler);
        }
        Float sceneRadius    = norm(scene->bbox().center() - scene->bbox().max);
        Float m_globalLookupRadius = m_globalLookupRadiusRelative * sceneRadius;
        Float m_causticLookupRadius = m_causticLookupRadiusRelative * sceneRadius;

        SurfaceInteraction3f si = scene->ray_intersect(ray);

        Spectrum LiSurf(0.0f), LiMedium(0.0f), transmittance(1.0f);
        int maxDepth = m_maxDepth == -1 ? INT_MAX : (m_maxDepth); // - rRec.depth)
        if (si.is_valid())
            LiSurf += m_globalPhotonMap->estimateIrradiance(
                si.p, si.n, m_globalLookupRadius, maxDepth, m_globalLookupSize);// * bsdf->getDiffuseReflectance(its) * math::InvPi;

        return { LiSurf, si.is_valid() };
    }

    void preProcess(const Scene *scene, Sampler *sampler) const {
        Log(LogLevel::Info, "Pre Processing Photon Map...");
        const int n = 100;
        // 1. For each light source in the scene we create a set of photons
        //    and divide the overall power of the light source amongst them.       

        for (int i = 0; i < n; i++) {
            MediumInteraction3f mRec;
            SurfaceInteraction3f si;
            Interaction3f its;
            ref<Sensor> sensor = scene->sensors()[0];
            Float time =
                sensor->shutter_open() + 0.5f * sensor->shutter_open_time();
            PositionSample3f pRec();

            for (int index = 0; index < n; ++index) {
                sampler->seed(index);
                EmitterPtr emitter;
                const Medium *medium;
                Spectrum power;
                RayDifferential3f ray;

                auto tuple = sample_emitter_direction(
                    scene, its, sampler->next_2d(), false, true);
                power   = std::get<1>(tuple);
                emitter = std::get<2>(tuple);
                medium  = emitter->medium();

                int depth = 1, nullInteractions = 0;
                bool delta               = false;
                bool lastNullInteraction = false;

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
                    } else if (its.t ==
                               std::numeric_limits<Float>::infinity()) {
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
                        handleSurfaceInteraction(depth, nullInteractions, delta,
                                                 si, medium,
                                                 throughput * power);
                        BSDFContext bCtx(TransportMode::Importance);
                        std::pair<BSDFSample3f, Spectrum> _sample =
                            bsdf->sample(bCtx, si, sampler->next_1d(),
                                         sampler->next_2d());
                        BSDFSample3f bsdfSample = _sample.first;
                        Spectrum bsdfWeight     = _sample.second;
                        if (bsdfWeight == Spectrum(0.0f))
                            break;

                        /* Prevent light leaks due to the use of shading normals
                         * * -- [Veach, p. 158] */
                        Vector3f wi = -ray.d, wo = si.to_world(bsdfSample.wo);
                        Float wiDotGeoN = dot(si.n, wi),
                              woDotGeoN = dot(si.n, wo);
                        if (wiDotGeoN * Frame3f::cos_theta(-bsdfSample.wo) <=
                                0 ||
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
                            throughput *=
                                std::abs((Frame3f::cos_theta(-bsdfSample.wo) *
                                          woDotGeoN) /
                                         (Frame3f::cos_theta(bsdfSample.wo) *
                                          wiDotGeoN));
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
            } else {
                ScalarFloat emitter_pdf = 1.f / scene->emitters().size();
                UInt32 index            = min(
                    UInt32(sample.x() * (ScalarFloat) scene->emitters().size()),
                    (uint32_t) scene->emitters().size() - 1);
                sample.x() = (sample.x() - index * emitter_pdf) *
                             scene->emitters().size();
                emitter =
                    gather<EmitterPtr>(scene->emitters().data(), index, active);
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
        m_globalPhotonMap->insert(
            Photon(si.p, si.n, -si.to_world(si.wi), weight, depth));
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
