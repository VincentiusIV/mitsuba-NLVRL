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
#include <mitsuba/core/timer.h>
#include <random>
#include "photonmap.h"

NAMESPACE_BEGIN(mitsuba)

const int k          = 3;

template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef typename PhotonMap::PhotonData PhotonData;

    PhotonMapper(const Properties &props) : Base(props) {
        m_globalPhotons     = props.int_("global_photons", 1000);
        m_causticPhotons    = props.int_("caustic_photons", 1000);
        m_volumePhotons     = props.int_("volume_photons", 1000);
        m_globalPhotonMap   = new PhotonMap(m_globalPhotons);
        m_causticPhotonMap  = new PhotonMap(m_causticPhotons);
        m_volumePhotonMap   = new PhotonMap(m_volumePhotons);

        m_rrDepth = props.int_("rr_start_depth", 5);
        m_maxDepth = props.int_("max_depth", 512);
        
        m_volumeLookupRadiusRelative = props.float_("volume_lookup_radius_relative", 0.01f);
        m_globalLookupRadiusRelative = props.float_("global_lookup_radius_relative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("caustic_lookup_radius_relative", 0.0125f);
        m_globalLookupSize = props.int_("global_lookup_size", 120);
        m_causticLookupSize = props.int_("caustic_lookup_size", 120);
        m_volumeLookupSize = props.int_("volume_lookup_size", 120);

        m_stochasticGather            = props.bool_("stochastic_gather", true);
        m_useNonLinear                = props.bool_("use_non_linear", true);
        m_useNonLinearCameraRays      = props.bool_("use_non_linear_camera", true);
        m_useFirstPhoton              = props.bool_("use_first_photon", false);
        m_directOnly                  = props.bool_("direct_only", false);

        m_useLaser                    = props.bool_("use_laser", false);
        laserOrigin                   = props.vector3f("laser_origin", Vector3f());
        laserDirection                = props.vector3f("laser_direction", Vector3f());
    }

    void preprocess(Scene* scene, Sensor* sensor) override {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        m_preprocessTimer.reset();

        for each (auto shape in scene->shapes()) {
            if (shape->interior_medium() != nullptr)
            {
                ScalarBoundingBox3f shape_bbox = shape->bbox();
                shape->build(shape_bbox.min, shape_bbox.max);
            }
        }

        Sampler *sampler = sensor->sampler();
        sampler->seed(0);

        // ------------------- Debug Info -------------------------- //
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string photonString                  = "- Emitter Count: " + std::to_string(emitters.size());
        Log(LogLevel::Info, photonString.c_str());

        for (int i = 0; i < emitters.size(); i++) {
            std::string emitterType = "- E" + std::to_string(i) + " = " + typeid(&emitters[i]).name();
            Log(LogLevel::Info, emitterType.c_str());
        }

        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);
        Float time = 0;
        static int greatestDepth = 0;

        int surfacePathCount = 0, volumePathCount = 0;
        int count = 0;
        while (!m_globalPhotonMap->is_full() || !m_volumePhotonMap->is_full()) {
            if (count++ % 100000 == 0) {
                std::ostringstream oss;
                oss << "photons[global=" << string::indent(m_globalPhotonMap->size()) << ", caustic= " << m_causticPhotonMap->size() << ", volume=" << m_volumePhotonMap->size() << "]";
                Log(LogLevel::Info, oss.str().c_str());
            }
            bool surfacePath = false, volumePath = false;

            if (m_globalPhotonMap->is_full() && m_volumePhotonMap->size() == 0)
                break; // stop, no volume in this scene.

            if (m_volumePhotonMap->is_full() && m_globalPhotonMap->size() == 0)
                break; // stop no surfaces in this scene.

            sampler->advance();
            EmitterPtr emitter = nullptr;
            MediumPtr medium   = nullptr;
            Spectrum throughput(1.0f);
            MediumInteraction3f mi = zero<MediumInteraction3f>();
            mi.t                   = math::Infinity<Float>;
            SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
            si.t                    = math::Infinity<Float>;
            Medium::NonLinearInteraction nli;

            Ray3f ray;
            Spectrum flux;
            // Sample random emitter

            emitter = sampleEmitter(scene, sampler->next_2d(), true);
            if (neq(emitter, nullptr)) {
                auto rayColorPair = emitter->sample_ray(0.0, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
                ray               = rayColorPair.first;
                flux              = rayColorPair.second * M_PI * 0.5;
                medium = emitter->medium();
            }

            if (m_useLaser) {
                ray.o    = Point3f(laserOrigin);
                ray.d    = normalize(laserDirection);
                ray.mint = 0.0f;
                ray.maxt = math::Infinity<Float>;
                ray.update();
            }

            float eta(1.0f);
            int nullInteractions = 0, mediumDepth = 0;
            bool wasTransmitted = false;
            bool fromLight = true;

            //
            Mask active             = true;
            Mask needs_intersection = true;
            UInt32 depth = 1, channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
            }

            for (int bounce = 0;; ++bounce) {
                active &= any(neq(depolarize(throughput), 0.f));
                Float q         = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                Mask perform_rr = (depth > (uint32_t) m_rrDepth);
                active &= sampler->next_1d(active) < q || !perform_rr;
                masked(throughput, perform_rr) *= rcp(detach(q));

                Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
                if (none(active) || all(exceeded_max_depth))
                {
                    break;
                }

                // -------------------- RTE ----------------- //
                Mask active_medium  = active && neq(medium, nullptr);
                Mask active_surface = active && !active_medium;
                Mask act_null_scatter = false, act_medium_scatter = false, escaped_medium = false;
                #pragma region RTE
                
                Mask is_spectral  = active_medium;
                Mask not_spectral = false;
                if (any_or<true>(active_medium)) {
                    is_spectral &= medium->has_spectral_extinction();
                    not_spectral = !is_spectral && active_medium;
                }

                if (any_or<true>(active_medium)) {
                    mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);

                    if (m_useNonLinear && medium->is_nonlinear()) {
                        nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);
                        while (nli.t < mi.t && nli.is_valid) {
                            bool valid = medium->handleNonLinearInteraction(scene, sampler, nli, si, mi, ray, throughput, channel, active_medium);
                            if (!valid)
                                break;

                            nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);
                        }
                    }

                    masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                    Mask intersect = needs_intersection && active_medium;
                    if (any_or<true>(intersect))
                        masked(si, intersect) = scene->ray_intersect(ray, intersect);
                    needs_intersection &= !active_medium;

                    masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;
                    if (any_or<true>(is_spectral)) {
                        auto [tr, free_flight_pdf] = medium->eval_tr_and_pdf(mi, si, is_spectral);
                        Float tr_pdf               = index_spectrum(free_flight_pdf, channel);
                        masked(throughput, is_spectral) *= select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                    }

                    escaped_medium = active_medium && !mi.is_valid();
                    active_medium &= mi.is_valid();

                    // Handle null and real scatter events
                    Mask null_scatter = sampler->next_1d(active_medium) >= index_spectrum(mi.sigma_t, channel) / index_spectrum(mi.combined_extinction, channel);

                    act_null_scatter |= null_scatter && active_medium;
                    act_medium_scatter |= !act_null_scatter && active_medium;

                    if (any_or<true>(is_spectral && act_null_scatter))
                        masked(throughput, is_spectral && act_null_scatter) *= mi.sigma_n * index_spectrum(mi.combined_extinction, channel) / index_spectrum(mi.sigma_n, channel);

                    masked(depth, act_medium_scatter) += 1;
                }

                active &= depth < (uint32_t) m_maxDepth;
                act_medium_scatter &= active;

                if (any_or<true>(act_null_scatter)) {
                    masked(ray.o, act_null_scatter)    = mi.p;
                    masked(ray.mint, act_null_scatter) = 0.f;
                    masked(si.t, act_null_scatter)     = si.t - mi.t;
                }

                if (any_or<true>(act_medium_scatter)) {
                    if (any_or<true>(is_spectral))
                        masked(throughput, is_spectral && act_medium_scatter) *= mi.sigma_s * index_spectrum(mi.combined_extinction, channel) / index_spectrum(mi.sigma_t, channel);
                    if (any_or<true>(not_spectral))
                        masked(throughput, not_spectral && act_medium_scatter) *= mi.sigma_s / mi.sigma_t;

                    PhaseFunctionContext phase_ctx(sampler);
                    auto phase = mi.medium->phase_function();
                    if (!fromLight || m_useFirstPhoton || m_directOnly && mediumDepth == 0) {
                        volumePath |= handleMediumInteraction(depth - nullInteractions, wasTransmitted, mi.p, medium, -ray.d, flux * throughput);
                    }
                    fromLight = false;
                    ++mediumDepth;
                    // ------------------ Phase function sampling -----------------
                    masked(phase, !act_medium_scatter) = nullptr;
                    auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                    RayDifferential3f new_ray(mi.spawn_ray(wo));
                    new_ray.mint                       = 0.0f;
                    masked(ray, act_medium_scatter)    = new_ray;
                    needs_intersection |= act_medium_scatter;
                }

                #pragma endregion RTE

                // --------------------- Surface Interactions ---------------------
                active_surface |= escaped_medium;
                Mask intersect = active_surface && needs_intersection;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);
                active_surface &= si.is_valid();

                // -------------------- End RTE ----------------- //

                if (any_or<true>(active_surface)) {

                    if (si.shape->is_emitter())
                        break;

                    surfacePath |= handleSurfaceInteraction(ray, depth, wasTransmitted, si, medium, flux * throughput);
                    
                    BSDFContext bCtx;
                    BSDFPtr bsdf  = si.bsdf(ray);

                    auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                    bsdfVal            = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);
                    

                    throughput = throughput  * bsdfVal;
                    active &= any(neq(depolarize(throughput), 0.f));
                    if (none_or<false>(active)) {
                        break;
                    }
                    eta *= bs.eta;

                    Ray3f bsdf_ray(si.spawn_ray(si.to_world(bs.wo)));

                    masked(ray, active_surface) = bsdf_ray;
                    needs_intersection |= active_surface;

                    Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                    masked(depth, non_null_bsdf) += 1;
                    masked(nullInteractions, !non_null_bsdf) += 1;

                    valid_ray |= non_null_bsdf;
                    wasTransmitted = non_null_bsdf && (has_flag(bs.sampled_type, BSDFFlags::Transmission) || has_flag(bs.sampled_type, BSDFFlags::Reflection));

                    if (non_null_bsdf && (m_useFirstPhoton || !wasTransmitted))
                        fromLight = false;

                    Mask intersect2             = active_surface && needs_intersection;
                    SurfaceInteraction3f si_new = si;
                    if (any_or<true>(intersect2))
                        si_new = scene->ray_intersect(ray, active);
                    needs_intersection &= !intersect2;

                    Mask has_medium_trans            = active_surface && si.is_medium_transition();
                    masked(medium, has_medium_trans) = si.target_medium(ray.d);

                    si = si_new;
                }

                active &= (active_surface | active_medium);
            }

            if (surfacePath)
                ++surfacePathCount;
            if (volumePath)
                ++volumePathCount;
        }
        std::string desad = "total emissions = " + std::to_string(count);
        Log(LogLevel::Info, desad.c_str());

        float scale          = 1.0 / surfacePathCount;
        std::string debugStr = "Global Photon scale: " + std::to_string(scale);
        Log(LogLevel::Info, debugStr.c_str());

        if (m_globalPhotonMap->size() > 0) {
            m_globalPhotonMap->setScaleFactor(scale);
            m_globalPhotonMap->build();
            debugStr = "Building global PM, size: " + std::to_string(m_globalPhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
        } else {
            Log(LogLevel::Info, "No global photons");
        }


        if (m_causticPhotonMap->size() > 0) {
            m_causticPhotonMap->setScaleFactor(scale);
            debugStr = "Building caustic PM, size: " + std::to_string(m_causticPhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            m_causticPhotonMap->build();
        } else {
            Log(LogLevel::Info, "No caustic photons");
        }

        volumePreprocessTimer.reset();
        if (m_volumePhotonMap->size() > 0) {
            m_volumePhotonMap->setScaleFactor(1.0f / volumePathCount);
            debugStr = "Building volume PM, size: " + std::to_string(m_volumePhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            m_volumePhotonMap->build();
            //m_bre->build(m_volumePhotonMap, m_volumeLookupSize);
        } else {
            Log(LogLevel::Info, "No volume photons");
        }
        Log(Info, "Volume Map Constructed. (took %s)", util::time_string(volumePreprocessTimer.value(), true));
        Log(Info, "Pre-process finished. (took %s)", util::time_string(m_preprocessTimer.value(), true));
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &_ray,
                                     const Medium *_medium, Float *aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static Float m_globalLookupRadius = -1, m_causticLookupRadius = -1, m_volumeLookupRadius = -1;
        if (m_globalLookupRadius == -1) {
            Float sceneRadius =
                norm(scene->bbox().center() - scene->bbox().max);
            m_globalLookupRadius  = m_globalLookupRadiusRelative * sceneRadius;
            m_causticLookupRadius = m_causticLookupRadiusRelative * sceneRadius;
            m_volumeLookupRadius = m_volumeLookupRadiusRelative * sceneRadius;
            std::string lookupString = "- Volume Lookup Radius: " + std::to_string(m_volumeLookupRadius);
            Log(LogLevel::Info, lookupString.c_str());
            lookupString = "- Scene Radius: " + std::to_string(sceneRadius);
            Log(LogLevel::Info, lookupString.c_str());
        }

        Ray3f ray(_ray);
        MediumPtr medium(_medium);
        Spectrum radiance(0.0f), throughput(1.0f);

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = math::Infinity<Float>;
        MediumInteraction3f mi  = zero<MediumInteraction3f>();
        mi.t = math::Infinity<Float>;
        Medium::NonLinearInteraction nli;

        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

        float eta(1.0f);
        int nullInteractions = 0;
        bool delta     = false;

        Mask specular_chain     = active && !m_hide_emitters;
        Mask needs_intersection = true;
        UInt32 depth = 1, channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        for (int bounce = 0;; ++bounce) {

            if (none(active))
                break;

            // -------------------- RTE ----------------- //
            Mask active_medium    = active && neq(medium, nullptr);
            Mask active_surface   = active && !active_medium;
            Mask escaped_medium = false;

            if (bounce > 1000) {
                std::ostringstream stream;
                stream << "over 1k bounces, that cant be right: " << std::endl 
                    << "active: " << active << std::endl
                    << "active_medium: " << active_medium << std::endl
                    << "active_surface: " << active_surface << std::endl
                    << "ray: " << ray << std::endl;
                std::string str = stream.str();
                Log(LogLevel::Error, str.c_str());
            }

#pragma region RTE
            Mask is_spectral  = active_medium;
            Mask not_spectral = false;
            if (any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (any_or<true>(active_medium)) {

                Float radius = m_volumeLookupRadius * enoki::lerp(0.5f, 1.5f, sampler->next_1d());
         
                Ray3f mediumRay(ray);
                mediumRay.mint = 0.0f;
                mediumRay.maxt = radius;
            
                size_t MVol = 0;
                size_t M    = 0;
                Spectrum volRadiance(0.0f);

                // Gather along the entire ray
                if (si.is_valid()) {
                    Timer volumeQueryTimer;
                    volumeQueryTimer.reset();

                    Spectrum gatherThroughput = throughput;
                    Ray3f gatherRay(ray);
                    gatherRay.maxt = si.t;

                    int localGatherCount = 0;
                    while (mediumRay.maxt < si.t) {
                        ++localGatherCount;
                        if (localGatherCount > 100000) {
                            Log(LogLevel::Warn, "prob endless loop");
                            break;
                        }

                        if (m_useNonLinearCameraRays && m_useNonLinear && medium->is_nonlinear()) {
                            nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);
                            while (nli.t <= mediumRay.maxt && nli.is_valid) {
                                                            
                                bool valid = medium->handleNonLinearInteraction(scene, sampler, nli, si, mi, ray, throughput, channel, active_medium);
                                if (!valid)
                                    break;

                                float gatherMaxt = mediumRay.maxt - nli.t;
                                mediumRay = Ray3f(ray);
                                mediumRay.maxt = gatherMaxt;

                                nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);
                            }
                        }

                        throughput *= medium->evalTransmittance(mediumRay, sampler, active);
                        Point3f gatherPoint = mediumRay(mediumRay.maxt);
                        Spectrum estimate = m_volumePhotonMap->estimateRadianceVolume(gatherPoint, mediumRay.d, medium, sampler, radius, M);

                        estimate *= gatherThroughput;

                        MVol += M;
                        volRadiance += estimate;

                        si.t -= mediumRay.maxt;
                        mediumRay.o = gatherPoint;
                        mediumRay.maxt = radius * 2.0f;
                    }

                    volRadiance *= m_volumePhotonMap->getScaleFactor();
                    radiance += volRadiance;

                    throughput *= medium->evalTransmittance(gatherRay, sampler, active);

                    ++volumeQueryCount;
                    gatherCount += localGatherCount; 
                    volumeQueryTime += volumeQueryTimer.value();
                   
                }

                MVol += M;

                // Sample medium interaction to see if we can continue
                mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;
                if (mi.is_valid())
                    break;
                
                escaped_medium = true;
                needs_intersection= true;
                active_surface |= si.is_valid();
            }

            active &= depth < (uint32_t) m_maxDepth;

#pragma endregion

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);

            if (any_or<true>(active_surface)) {
                // ---------------- Intersection with emitters ----------------
                EmitterPtr emitter            = si.emitter(scene);
                Mask use_emitter_contribution = active_surface && specular_chain && neq(emitter, nullptr);
                if (any_or<true>(use_emitter_contribution))
                    masked(radiance, use_emitter_contribution) += throughput * emitter->eval(si, use_emitter_contribution);
            }

            active_surface &= si.is_valid();

            // -------------------- End RTE ----------------- //

            if (any_or<true>(active_surface)) {
                if (si.shape->is_emitter())
                    break;
                
                BSDFContext bCtx;
                BSDFPtr bsdf = si.bsdf(ray);

                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth);
                // && !has_flag(bs.sampled_type, BSDFFlags::Reflection);
                // Photon Map Sampling
                if (likely(any_or<true>(active_e))) {
                    Timer surfaceQueryTimer;
                    surfaceQueryTimer.reset();

                    radiance[active_surface] += m_causticPhotonMap->estimateCausticRadiance(si, m_causticLookupRadius, m_causticLookupSize) * throughput;
                    radiance[active_surface] += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize) * throughput;

                    
                    ++surfaceQueryCount;
                    surfaceQueryTime += surfaceQueryTimer.value();
                    
                    break;
                }

                auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                bsdfVal            = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

                throughput = throughput * bsdfVal;
                active &= any(neq(depolarize(throughput), 0.f));
                if (none_or<false>(active)) {
                    break;
                }
                eta *= bs.eta;

                RayDifferential3f bsdf_ray(si.spawn_ray(si.to_world(bs.wo)));

                masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                masked(depth, non_null_bsdf) += 1;
                masked(nullInteractions, !non_null_bsdf) += 1;

                valid_ray |= non_null_bsdf;
                specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));

                delta = non_null_bsdf && (has_flag(bs.sampled_type, BSDFFlags::Transmission) || has_flag(bs.sampled_type, BSDFFlags::Reflection));

                Mask intersect2             = active_surface && needs_intersection;
                SurfaceInteraction3f si_new = si;
                if (any_or<true>(intersect2))
                    si_new = scene->ray_intersect(ray, active);
                needs_intersection &= !intersect2;

                Mask has_medium_trans            = active_surface && si.is_medium_transition();
                masked(medium, has_medium_trans) = si.target_medium(ray.d);

                si = si_new;
            }

            active &= (active_surface | active_medium);
        }

        return { radiance, valid_ray };
    }

    EmitterPtr sampleEmitter(const Scene *scene, const Point2f &sample_, Mask active) const {
        Point2f sample(sample_);
        EmitterPtr emitter = nullptr;

        if (likely(!scene->emitters().empty())) {
            if (scene->emitters().size() == 1) {
                emitter = scene->emitters()[0];
            } else {
                ScalarFloat emitter_pdf = 1.f / scene->emitters().size();
                UInt32 index = min(UInt32(sample.x() * (ScalarFloat) scene->emitters().size()), (uint32_t) scene->emitters().size() - 1);
                sample.x() = (sample.x() - index * emitter_pdf) * scene->emitters().size();
                emitter = scene->emitters()[index];
            }
        }

        return emitter;
    }

    void postprocess(Scene *scene, Sensor *sensor) override {

        std::ostringstream stream;
        stream << "Surface Query Count: " << surfaceQueryCount << std::endl
               << "Surface Query Time: " <<  util::time_string(surfaceQueryTime) << std::endl
               << "Volume Query Count: " <<  volumeQueryCount << std::endl
               << "Volume Gather Count: " <<  gatherCount << std::endl
               << "Volume Query Time: " << util::time_string(volumeQueryTime) << std::endl
               << "Global Map Size: " << util::mem_string(m_globalPhotonMap->getSize()) << std::endl
               << "Caustic Map Size: " << util::mem_string(m_causticPhotonMap->getSize()) << std::endl
               << "Volume Map Size: " << util::mem_string(m_volumePhotonMap->getSize()) << std::endl;
        std::string str = stream.str();
        Log(LogLevel::Info, str.c_str());
    }

    bool handleSurfaceInteraction(const Ray3f &ray, int depth, bool wasTransmitted,
                                  const SurfaceInteraction3f &si,
                                  const Medium *medium,
                                  const Spectrum &weight) const {
        BSDFPtr bsdf       = si.bsdf();
        uint32_t bsdfFlags = bsdf->flags();
        if (!has_flag(bsdf->flags(), BSDFFlags::Smooth))
            return false;
        if (!wasTransmitted) {
            return m_globalPhotonMap->insert(si.p, PhotonData(si.n, ray.d, weight, depth));
        } else {
            return m_causticPhotonMap->insert(si.p, PhotonData(si.n, ray.d, weight, depth));
        }
    }

    bool handleMediumInteraction(int depth, bool delta,
                                 const Point3f &p,
                                 const Medium *medium, const Vector3f &wi,
                                 const Spectrum &weight) const {
        return m_volumePhotonMap->insert(p, PhotonData(Normal3f(0.0f, 0.0f, 0.0f), -wi, weight, depth));
    }

    MTS_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m, eq(idx, 1u)) = spec[1];
            masked(m, eq(idx, 2u)) = spec[2];
        } else {
            ENOKI_MARK_USED(idx);
        }
        return m;
    }

    MTS_DECLARE_CLASS()
private:
    PhotonMap *m_globalPhotonMap;
    PhotonMap *m_causticPhotonMap;
    PhotonMap *m_volumePhotonMap;

    Vector3f laserOrigin, laserDirection;

    int m_rrDepth, m_maxDepth;
    int m_minDepth = 1;
    int m_globalPhotons, m_causticPhotons, m_volumePhotons;
    float m_globalLookupRadiusRelative, m_causticLookupRadiusRelative, m_volumeLookupRadiusRelative;
    int m_globalLookupSize, m_causticLookupSize, m_volumeLookupSize;

    bool m_useNonLinear;
    bool m_useNonLinearCameraRays;
    bool m_useLaser;
    bool m_useFirstPhoton;
    bool m_directOnly;
    bool m_stochasticGather;

    mutable std::atomic<int> surfaceQueryCount, volumeQueryCount, gatherCount;
    mutable std::atomic<size_t> surfaceQueryTime, volumeQueryTime;

    Timer m_preprocessTimer, volumePreprocessTimer;
};

MTS_IMPLEMENT_CLASS_VARIANT(PhotonMapper, SamplingIntegrator);
MTS_EXPORT_PLUGIN(PhotonMapper, "Photon Mapping integrator");
NAMESPACE_END(mitsuba)
