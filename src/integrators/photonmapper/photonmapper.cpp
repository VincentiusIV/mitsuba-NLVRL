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
const int numPhotons = 10000;

template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef typename PhotonMap::PhotonData PhotonData;

    PhotonMapper(const Properties &props) : Base(props) {
        m_globalPhotonMap    = new PhotonMap(numPhotons);
        m_causticPhotonMap = new PhotonMap(numPhotons);
        m_volumePhotonMap    = new PhotonMap(numPhotons);


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

    void preprocess(Scene* scene, Sensor* sensor) const override {
        static bool m_isPreProcessed = false;
        Log(LogLevel::Info, "Pre Processing...");     
 
        if (!m_isPreProcessed) {
            m_isPreProcessed = true;
            preProcess(scene, sensor->sampler()->clone());
        }
        Log(LogLevel::Info, "Pre Processing done.");     
    }

    Spectrum E(const Scene *scene, const SurfaceInteraction3f &si,
               const Medium *medium, Sampler *sampler, int nSamples,
               bool handleIndirect) const {

        Spectrum E(0.0f);
        Frame3f frame(si.sh_frame);
        int depth = 1;
        sampler->seed(0);
        for (int i = 0; i < nSamples; i++) {

            // indirect illum
            Vector3f dir = frame.to_world(
                warp::square_to_cosine_hemisphere(sampler->next_2d()));
            RayDifferential3f indirectRay =
                RayDifferential3f(si.p, dir, si.time);
            SurfaceInteraction3f indirectSi = scene->ray_intersect(indirectRay);
            ++depth;
            E += Li(indirectRay, indirectSi, scene, sampler->clone(), depth);

            //sampler->advance();
        }

        return E / nSamples;
    }

    Spectrum Li(const RayDifferential3f &ray, const SurfaceInteraction3f &si,
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

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel =
                (UInt32) min(sampler->next_1d() * n_channels, n_channels - 1);
        }
        

        MediumPtr medium    = si.target_medium(ray.d);
        Mask active_medium = neq(medium, nullptr);
        if (any_or<true>(active_medium)) {

            Ray mediumRaySegment(ray, 0, si.t);
            Mask is_spectral       = active_medium;
            MediumInteraction3f mi = medium->sample_interaction(
                ray, sampler->next_1d(), channel, active_medium);
            auto [tr, pdf] = medium->eval_tr_and_pdf(mi, si, is_spectral);
            transmittance         = tr;
            mediumRaySegment.mint = ray.mint;
            //if ((depth < m_maxDepth || m_maxDepth < 0) && m_bre.get() != NULL)
            //    LiMedium = m_bre->query(mediumRaySegment, rRec.medium, m_maxDepth - 1);
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

        if (hasSpecular && exhaustiveSpecular) {
            /* 1. Specular indirect */
            size_t compCount = bsdf->component_count();
            for (size_t i = 0; i < compCount; i++) {
                if (!has_flag(bsdf->flags(), BSDFFlags::Delta))
                    continue;
                /* Sample the BSDF and recurse */
                BSDFContext ctx(TransportMode::Radiance);
                ctx.component = i;
                auto [ bsdfSample, bsdfVal] = bsdf->sample(ctx, si, 0.0f, Point2f(0.5f));
                if (bsdfVal == Spectrum(0.0f))
                    continue;

                // parent integrator unsupported
                RayDifferential3f bsdfRay(si.p, si.to_world(bsdfSample.wo), ray.time);
                SurfaceInteraction3f bsdfSi = scene->ray_intersect(bsdfRay);
                ++depth;
                LiSurf += bsdfVal * Li(bsdfRay, bsdfSi, scene, sampler->clone(), depth);
            }
        }

        /* Estimate the direct illumination if this is requested */
        int numEmitterSamples = m_directSamples, numBSDFSamples;
        Float weightLum, weightBSDF;
        std::vector<Point2f> sampleArray;
        
        if (depth > 1) {
            numBSDFSamples = numEmitterSamples = 1;
            weightLum = weightBSDF = 1.0f;
        } else {
            if (isDiffuse) {
                numBSDFSamples = m_directSamples;
                weightBSDF = weightLum = m_invEmitterSamples;
            } else {
                numBSDFSamples = m_glossySamples;
                weightLum      = m_invEmitterSamples;
                weightBSDF     = m_invGlossySamples;
            }
        }

        if (has_flag(bsdf->flags(), BSDFFlags::Smooth)) {
            BSDFContext bCtx;
            auto [bs, bsdf_val] =
                bsdf->sample(bCtx, si, sampler->next_1d(),
                             sampler->next_2d(), true);
            bsdf_val       = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            Spectrum value = m_globalPhotonMap->estimateRadiance(
                                 si, m_globalLookupRadius, m_globalLookupSize) *
                             bsdf_val;
            LiSurf += value;
        }
        
        /* Sample direct compontent via BSDF sampling if this is generally
         requested AND the BSDF is smooth, or there is a delta component that
         was not handled by the exhaustive sampling loop above */
        bool bsdfSampleDirect = has_flag(bsdf->flags(), BSDFFlags::Smooth) ||
             (hasSpecular && !exhaustiveSpecular);

        /* Sample indirect component via BSDF sampling if this is generally
           requested AND the BSDF is non-diffuse (diffuse is handled by the
           global photon map) or there is a delta component that was not handled
           by the exhaustive sampling loop above. */
        bool bsdfSampleIndirect = !isDiffuse && (has_flag(bsdf->flags(), BSDFFlags::Smooth) || (hasSpecular && !exhaustiveSpecular));

        if (bsdfSampleDirect || bsdfSampleIndirect) {
            if (numBSDFSamples > 1) {
                int samples = std::max(m_directSamples, m_glossySamples);
                for (size_t i = 0; i < samples; i++) {
                    sampleArray.push_back(sampler->next_2d());
                }
            } else {
                sampleArray.push_back(sampler->next_2d());
            }

            for (int i = 0; i < numBSDFSamples; ++i) {
                /* Sample BSDF * cos(theta) */
                BSDFContext bCtx(TransportMode::Radiance);
                auto [bsdfSample, bsdfVal] = bsdf->sample(bCtx, si, 0, sampleArray[i]);
                if (bsdfVal == Spectrum(0.0f))
                    continue;

                /* Trace a ray in this direction */
                RayDifferential3f bsdfRay(si.p, si.to_world(bsdfSample.wo), ray.time);
                SurfaceInteraction3f bsdfSi = scene->ray_intersect(bsdfRay);
                Spectrum value;
                bool hitEmitter = false;
                if (bsdfSi.is_valid()) {
                    if (bsdfSi.emitter(scene) != nullptr && bsdfSampleDirect) {
                        value      = bsdfSi.emitter(scene)->eval(bsdfSi);
                        hitEmitter = true;
                    }
                } else if (bsdfSampleDirect) {
                    const Emitter *env = scene->environment();

                    if (env != nullptr) {
                        value = env->eval(bsdfSi);
                        hitEmitter = true;
                    }
                }

                if (hitEmitter) {
                    EmitterPtr emitter = bsdfSi.emitter(scene);
                    DirectionSample3f ds(bsdfSi, si);
                    ds.object = emitter;

                    Float emitterPdf = scene->pdf_emitter_direction(bsdfSi, ds);
                    Spectrum transmittance(1.0f);
                    if (bsdfSi.target_medium(bsdfRay.d)) {
                        MediumInteraction3f mi = medium->sample_interaction(bsdfRay,0, channel, true);
                        std::pair<UnpolarizedSpectrum, UnpolarizedSpectrum> tr_and_pdf =
                                bsdfSi.target_medium(bsdfRay.d)->eval_tr_and_pdf(mi, bsdfSi, true);
                        transmittance = tr_and_pdf.first;
                    }
                    const Float weight = miWeight(bsdfSample.pdf * numBSDFSamples,
                                 emitterPdf * numEmitterSamples) * weightBSDF;
                    LiSurf += value * bsdfVal * weight * transmittance;
                }

                /* Recurse */
                if (bsdfSampleIndirect) {
                    LiSurf += bsdfVal * Li(bsdfRay, bsdfSi, scene, sampler->clone(), depth) * weightBSDF;
                }
            }
        }
       
        return LiSurf * transmittance + LiMedium;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
        RayDifferential3f r(ray);
        SurfaceInteraction3f si = scene->ray_intersect(ray);
        Spectrum e_value(0.0f);
        int maxDepth = m_maxDepth == -1 ? INT_MAX : (m_maxDepth);

        e_value += E(scene, si, medium, sampler, 16, true);

        return { e_value, active };
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

    void preProcess(const Scene *scene, Sampler *sampler) const {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        sampler->seed(0);

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

        MediumInteraction3f mi;
        SurfaceInteraction3f si;
        Interaction3f its;
        ref<Sensor> sensor = scene->sensors()[0];
        Float time = sensor->shutter_open() + 0.5f * sensor->shutter_open_time();

        int numShot = 0;

        for (int index = 0; index < numPhotons; index++) {
            std::string debugStr =  "- Photon Num: " + std::to_string(index);
            Log(LogLevel::Info, debugStr.c_str());   
            
            EmitterPtr emitter;
            MediumPtr medium;
            Spectrum power(1.0f);

            // Sample random emitter
            auto tuple = sample_emitter_direction(scene, its, sampler->next_2d(), false, true);
            emitter = std::get<2>(tuple);

            auto rayColorPair = emitter->sample_ray(time, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            RayDifferential3f ray(rayColorPair.first);
            //power = rayColorPair.second;

            int depth = 1, nullInteractions = 0;
            bool delta               = false;
            bool lastNullInteraction = false;

            Spectrum throughput(1.0f);
            Mask active             = true;
            Mask needs_intersection = true;

            UInt32 channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
            }

            while (throughput != Spectrum(0.0f) && (depth <= m_maxDepth || m_maxDepth < 0)) {
                ++numShot;
                si = scene->ray_intersect(ray);

                 if (!si.is_valid()) {
                    break;
                 }

                 if (false) {

                 bool is_medium_trans = any_or<true>(si.is_medium_transition());
                 medium = si.target_medium(ray.d);
                 
                 Mask active_medium    = active && neq(medium, nullptr);
                 Mask active_surface   = active && !active_medium;
                 Mask act_null_scatter = false, act_medium_scatter = false,
                      escaped_medium = false;

                 // If the medium does not have a spectrally varying
                 // extinction, we can perform a few optimizations to speed
                 // up rendering
                 Mask is_spectral  = active_medium;
                 Mask not_spectral = false;
                 if (any_or<true>(active_medium)) {
                     is_spectral &= medium->has_spectral_extinction();
                     not_spectral = !is_spectral && active_medium;
                 }

                 if (any_or<true>(active_medium)) {
                     mi = medium->sample_interaction(
                         ray, sampler->next_1d(active_medium), channel,
                         active_medium);
                     masked(ray.maxt, active_medium &&
                                          medium->is_homogeneous() &&
                                          mi.is_valid()) = mi.t;
                     Mask intersect = needs_intersection && active_medium;
                     if (any_or<true>(intersect))
                         masked(si, intersect) =
                             scene->ray_intersect(ray, intersect);
                     needs_intersection &= !active_medium;

                     masked(mi.t, active_medium && (si.t < mi.t)) =
                         math::Infinity<Float>;
                     if (any_or<true>(is_spectral)) {
                         auto [tr, free_flight_pdf] =
                             medium->eval_tr_and_pdf(mi, si, is_spectral);
                         Float tr_pdf =
                             index_spectrum(free_flight_pdf, channel);
                         masked(throughput, is_spectral) *=
                             select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                     }

                     escaped_medium = active_medium && !mi.is_valid();
                     active_medium &= mi.is_valid();

                     // Handle null and real scatter events
                     Mask null_scatter =
                         sampler->next_1d(active_medium) >=
                         index_spectrum(mi.sigma_t, channel) /
                             index_spectrum(mi.combined_extinction, channel);

                     act_null_scatter |= null_scatter && active_medium;
                     act_medium_scatter |= !act_null_scatter && active_medium;

                     if (any_or<true>(is_spectral && act_null_scatter))
                         masked(throughput, is_spectral && act_null_scatter) *=
                             mi.sigma_n *
                             index_spectrum(mi.combined_extinction, channel) /
                             index_spectrum(mi.sigma_n, channel);

                     masked(depth, act_medium_scatter) += 1;

                     handleMediumInteraction(depth, nullInteractions, delta, mi,
                                             medium, -ray.d, throughput);
                 }

                 // Dont estimate lighting if we exceeded number of bounces
                 active &= depth < (uint32_t) m_maxDepth;
                 act_medium_scatter &= active;

                 if (any_or<true>(act_null_scatter)) {
                     masked(ray.o, act_null_scatter)    = mi.p;
                     masked(ray.mint, act_null_scatter) = 0.f;
                     masked(si.t, act_null_scatter)     = si.t - mi.t;
                 }

                 if (any_or<true>(act_medium_scatter)) {
                     if (any_or<true>(is_spectral))
                         masked(throughput,
                                is_spectral && act_medium_scatter) *=
                             mi.sigma_s *
                             index_spectrum(mi.combined_extinction, channel) /
                             index_spectrum(mi.sigma_t, channel);
                     if (any_or<true>(not_spectral))
                         masked(throughput,
                                not_spectral && act_medium_scatter) *=
                             mi.sigma_s / mi.sigma_t;

                     PhaseFunctionContext phase_ctx(sampler);
                     auto phase = mi.medium->phase_function();

                     // ------------------ Phase function sampling
                     // -----------------
                     masked(phase, !act_medium_scatter) = nullptr;
                     auto [wo, phase_pdf]               = phase->sample(
                         phase_ctx, mi, sampler->next_2d(act_medium_scatter),
                         act_medium_scatter);
                     Ray3f new_ray                   = mi.spawn_ray(wo);
                     new_ray.mint                    = 0.0f;
                     masked(ray, act_medium_scatter) = new_ray;
                     needs_intersection |= act_medium_scatter;
                 }
                 }
                 {
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
                    BSDFContext bCtx(TransportMode::Radiance);
                    auto [bs, bsdfWeight] = bsdf->sample(bCtx, si, sampler->next_1d(), sampler->next_2d());
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
                        throughput *= std::abs(
                            (Frame3f::cos_theta(-bs.wo) * woDotGeoN) /
                            (Frame3f::cos_theta(bs.wo) * wiDotGeoN));
                    }

                    handleSurfaceInteractionScattering(bs, si, ray);
                }

                if (depth++ >= m_rrStartDepth) {
                    Float q = enoki::min(enoki::hmax(throughput), (Float) 0.95f);
                    if (sampler->next_1d() >= q)
                        break;
                    throughput /= q;
                }
            }
        }

        m_globalPhotonMap->setScaleFactor(1.0f / numShot);
        m_globalPhotonMap->build();

        if (m_volumePhotonMap->size() > 0)
        {
            m_volumePhotonMap->setScaleFactor(1.0f / numShot);
            m_volumePhotonMap->build();
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
