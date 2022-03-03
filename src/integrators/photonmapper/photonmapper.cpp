#pragma once
#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include "kdtree.h"

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
            point[0] = position[0];
            point[1] = position[1];
            point[2] = position[2];
            this->spectrum = spectrum;
            this->normal   = normal;
            this->direction = direction;
        }
    };

    class PhotonMap : PointKDTree<Photon> {
    public:
        PhotonMap() { 
            Log(LogLevel::Info, "Constructing PhotonMap...");
        }

        inline void insert(const Photon &photon) { 
               
        }
    };

    PhotonMapper(const Properties &props) : Base(props) {
        m_globalPhotonMap = new PhotonMap();
        m_directSamples = props.int_("directSamples", 16);
        m_glossySamples = props.int_("glossySamples", 32);
        m_rrStartDepth  = props.int_("rrStartDepth", 5);
        m_maxDepth      = props.int_("maxDepth", 128);
        m_maxSpecularDepth = props.int_("maxSpecularDepth", 4);
        m_granularity      = props.int_("granularity", 0);
        m_globalPhotons    = props.int_("globalPhotons", 250000);
        m_causticPhotons   = props.int_("causticPhotons", 250000);
        m_volumePhotons    = props.int_("volumePhotons", 250000);
        m_globalLookupRadiusRelative =
            props.float_("globalLookupRadiusRelative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize = props.int_("globalLookupSize", 120);
        m_causticLookupSize = props.int_("causticLookupSize", 120);
        m_volumeLookupSize  = props.int_("volumeLookupSize", 120);
        m_gatherLocally     = props.bool_("gatherLocally", true);
        m_autoCancelGathering = props.bool_("autoCancelGathering", true);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium * medium,
                                     Float * aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static bool m_isPreProcessed = false;
        if (!m_isPreProcessed) {
            m_isPreProcessed = true;
            preProcess(scene, sampler);
        }

        Spectrum LiSurf(0.0f), LiMedium(0.0f), transmittance(1.0f);

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = math::Infinity<Float>;

        return { 
            LiSurf, si.is_valid() 
        };
    }

    void preProcess(const Scene *scene, Sampler *sampler) const {
        Log(LogLevel::Info, "Pre Processing Photon Map...");
        const int n = 100000;
        // 1. For each light source in the scene we create a set of photons
        //    and divide the overall power of the light source amongst them.
        
         /*for (int i = 0; i < n; i++) {
            MediumInteraction3f mRec;
            Interaction3f its;
            ref<Sensor> sensor = scene->sensors()[0];
            Point2f sample;
            Float time =
                sensor->shutter_open() + 0.5f * sensor->shutter_open_time();
            PositionSample3f pRec();

            for (int index = 0; index < n; ++index) {
                sampler->seed(index);
                EmitterPtr emitter;
                Medium *medium;
                Spectrum power;
                RayDifferential3f ray;

                // power = m_scene->sampleEmitterRay(ray, emitter,
                // m_sampler->next_2d(), m_sampler->next_2d(), pRec.time); auto
                // pair = sample_emitter_direction(scene, its,
                // sampler->next_2d(), false, true);

                // medium = emitter->medium();

                int depth = 1, nullInteractions = 0;
                bool delta               = false;
                bool lastNullInteraction = false;

                Spectrum throughput(1.0f);
                while (!throughput.isZero() &&
                       (depth <= m_maxDepth || m_maxDepth < 0)) {
                    si = scene->ray_intersect(photonRayDiff, true);
                    if (medium && medium->sampleDistance(Ray(ray, 0, its.t),
                mRec, m_sampler)) { throughput *= mRec.sigma_s *
                mRec.transmittance / mRec.pdfSuccess;
                        handleMediumInteraction(depth, nullInteractions, delta,
                mRec, medium, -ray.d, throughput * power);
                        PhaseFunctionSamplingRecord pRec(mRec, -ray.d,
                TransportMode::Importance); throughput *=
                medium->getPhaseFunction()->sample(pRec, m_sampler); delta =
                false; lastNullInteraction = false;
                        handleMediumInteractionScattering(mRec, pRec, ray);
                    } else if (its.t == std::numeric_limits<Float>::infinity())
                { break; } else { if (medium) throughput *= mRec.transmittance /
                mRec.pdfFailure; const BSDF *bsdf = its.bsdf();
                        handleSurfaceInteraction(depth, nullInteractions, delta,
                its, medium, throughput * power); BSDFSamplingRecord bRec(its,
                m_sampler, TransportMode::Importance); Spectrum bsdfWeight =
                bsdf->sample(bRec, sampler->next_2d()); if (bsdfWeight.isZero())
                            break;
                        Vector3f wi = -ray.d, wo = its.toWorld(bRec.wo);
                        Float wiDotGeoN = dot(its.geoFrame.n, wi),
                              woDotGeoN = dot(its.geoFrame.n, wo);
                        if (wiDotGeoN * Frame::cosTheta(bRec.wi) <= 0 ||
                            woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                            break;
                        throughput *= bsdfWeight;
                        if (its.isMediumTransition())
                            medium = its.getTargetMedium(woDotGeoN);
                        if (bRec.sampledType & BSDFFlags::Null) {
                            ++nullInteractions;
                            lastNullInteraction = true;
                        } else {
                            delta = bRec.sampledType & BSDFFlags::Delta;
                            lastNullInteraction = false;
                        }

                        handleSurfaceInteractionScattering(bRec, ray);
                    }

                    if (depth++ >= m_rrStartDepth) {
                        Float q = std::min(throughput.max(), (Float) 0.95f);
                        if (sampler->next_1d() >= q)
                            break;
                        throughput /= q;
                    }

                    if (depth <= m_maxDepth || m_maxDepth < 0) {
                        handledBounce(depth, nullInteractions, delta, ray.o,
                                      medium, lastNullInteraction,
                                      throughput * power);

                        handleSetRayDifferential(ray);
                    }
                }
            }
        }*/
    }

   /*std::tuple<DirectionSample3f, Spectrum, EmitterPtr>
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
                                  const Spectrum &weight) {
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
                                 const Spectrum &weight) {
        int depth = _depth - nullInteractions;
        if (depth < m_minDepth) {
            return;
        }
        m_volumePhotonMap->insert(
            Photon(mi.p, Normal3f(0.0f, 0.0f, 0.0f), -wi, weight, depth));
    }*/ 

    //void handleMediumInteractionScattering(const MediumInteraction3f& mi, ) {}


    MTS_DECLARE_CLASS()
private:
    PhotonMap* m_globalPhotonMap;
    PhotonMap* m_causticPhotonMap;
    PhotonMap* m_volumePhotonMap;

    int m_directSamples, m_glossySamples, m_rrStartDepth, m_maxDepth,
        m_maxSpecularDepth, m_granularity;
    int m_minDepth = 1;
    int m_globalPhotons, m_causticPhotons, m_volumePhotons;
    float m_globalLookupRadiusRelative, m_causticLookupRadiusRelative;
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
