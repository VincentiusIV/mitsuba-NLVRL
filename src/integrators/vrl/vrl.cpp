#include <random>
#include <enoki/stl.h>
#include <enoki/fwd.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>

#include "../photonmapper/photonmap.h"
#include "vrl_struct.h"
#include "vrl_map.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class VRLIntegrator : public SamplingIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef VRL<Float, Spectrum> VRL;
    typedef VRLMap<Float, Spectrum> VRLMap;
    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef typename PhotonMap::PhotonData PhotonData;

    VRLIntegrator(const Properties &props) : Base(props) { 
        m_numLightEmissions = props.int_("lightEmissions", 1000000);
        m_maxDepth = props.int_("max_depth", 512);
        m_directSamples               = props.int_("directSamples", 16);
        m_glossySamples               = props.int_("glossySamples", 32);
        m_maxSpecularDepth            = props.int_("maxSpecularDepth", 4);
        m_granularity                 = props.int_("granularity", 0);
        m_globalPhotons               = props.int_("globalPhotons", 250000);
        m_causticPhotons              = props.int_("causticPhotons", 250000);
        m_targetVRLs = props.int_("targetVRLs", 1000);
        m_globalLookupRadiusRelative  = props.float_("globalLookupRadiusRelative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize            = props.int_("globalLookupSize", 120);
        m_causticLookupSize           = props.int_("causticLookupSize", 120);
        m_gatherLocally               = props.bool_("gatherLocally", true);
        m_autoCancelGathering         = props.bool_("autoCancelGathering", true);       

        m_nbParticules  = props.int_("nbParticules", 50);
        m_rrDepth      = props.int_("rrDepth", 5);
        /* Maximum number of passes to render. -1 renders until the process is stopped. */
        m_useLightCut = props.bool_("useLightCut", false);

        // VRL Options
        m_diceVRL             = props.int_("diceVRL", 1);
        m_longVRL             = props.bool_("longVRL", false);
        m_useUniformSampling  = props.bool_("useUniformSampling", false);
        m_RRVRL               = props.bool_("RRVRL", false);
        m_scaleRR             = props.float_("scaleRR", 0.5); // 2 meters before 5%
        m_nbSamplesRay        = props.int_("nbSamplesRay", 1);
        m_thresholdBetterDist = props.int_("thresholdBetterDist", 8);
        m_thresholdError      = props.float_("thresholdError", 0.02);
        m_shootCenter         = props.bool_("shootCenter", true);

        m_globalPhotonMap  = new PhotonMap(m_numLightEmissions);
        m_causticPhotonMap = new PhotonMap(m_numLightEmissions);
        Log(LogLevel::Info, "Constructing VRL Map...");
        m_vrlMap = new VRLMap(m_nbParticules);
    }

    
    void preprocess(Scene *scene, Sensor *sensor) override {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        if (is_rgb_v<Spectrum>)
            Log(LogLevel::Info, "rgb tho");
        if (is_monochromatic_v<Spectrum>)
            Log(LogLevel::Info, "monochromatic tho");
        if (is_spectral_v<Spectrum>)
            Log(LogLevel::Info, "spectral tho");
        if (is_polarized_v<Spectrum>)
            Log(LogLevel::Info, "polarized tho");

        Sampler *sampler = sensor->sampler();
        sampler->seed(0);

        // ------------------- Debug Info -------------------------- //
        std::vector<int> depthCounter;
        std::string numPhotonsStr = "- Photon Count: " + std::to_string(m_numLightEmissions);
        Log(LogLevel::Info, numPhotonsStr.c_str());
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string photonString                  = "- Emitter Count: " + std::to_string(emitters.size());
        Log(LogLevel::Info, photonString.c_str());

        for (int i = 0; i < emitters.size(); i++) {
            std::string emitterType = "- E" + std::to_string(i) + " = " + typeid(&emitters[i]).name();
            Log(LogLevel::Info, emitterType.c_str());
        }

        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

        static int greatestDepth = 0;
        ScalarFloat emitter_pdf  = 1.f / scene->emitters().size();

        for (int index = 0; index < m_numLightEmissions; index++) {
            sampler->advance();
            EmitterPtr emitter = nullptr;
            MediumPtr medium   = nullptr;
            Spectrum throughput(1.0f);
            MediumInteraction3f mi  = zero<MediumInteraction3f>();
            mi.t                    = math::Infinity<Float>;
            SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
            si.t                    = math::Infinity<Float>;

            // Sample random emitter
            emitter           = sampleEmitter(scene, sampler->next_2d(), true);
            auto rayColorPair = emitter->sample_ray(0.0, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            RayDifferential3f ray(rayColorPair.first);
            Spectrum flux = emitter->getUniformRadiance();
            flux *= math::Pi<float> * emitter->shape()->surface_area();
            medium = emitter->medium();

            if (medium != nullptr)
                Log(LogLevel::Info, "endpiont surrounded by medium!");

            float eta(1.0f);
            int nullInteractions = 0, mediumDepth = 0;
            bool delta = false;

            //
            Mask active             = true;
            Mask needs_intersection = true;
            UInt32 depth = 1, channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
            }

            si = scene->ray_intersect(ray, active);
            if (!si.is_valid())
                continue;
            /*Mask has_medium_trans            = si.is_valid() && si.is_medium_transition();
            masked(medium, has_medium_trans) = si.target_medium(ray.d);*/

            for (int bounce = 0;; ++bounce) {
                active &= any(neq(depolarize(throughput), 0.f));
                Float q         = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                Mask perform_rr = (depth > (uint32_t) m_rrDepth);
                active &= sampler->next_1d(active) < q || !perform_rr;
                masked(throughput, perform_rr) *= rcp(detach(q));

                Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
                if (none(active) || all(exceeded_max_depth))
                    break;

                // -------------------- RTE ----------------- //
                Mask active_medium    = active && neq(medium, nullptr);
                Mask active_surface   = active && !active_medium;
                Mask act_null_scatter = false, act_medium_scatter = false, escaped_medium = false;
#pragma region RTE

                Mask is_spectral  = active_medium;
                Mask not_spectral = false;
                if (any_or<true>(active_medium)) {
                    is_spectral &= medium->has_spectral_extinction();
                    not_spectral = !is_spectral && active_medium;
                }

                if (any_or<true>(active_medium)) {
                    ++mediumDepth;
                    mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
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

                    if (any_or<true>(escaped_medium))
                        mediumDepth = 0;

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
                    
                                        
                    if (any_or<true>(act_medium_scatter)) {
                        if (m_vrlMap->size() < m_targetVRLs) {
                            VRL vrl(ray.o, mi.medium, throughput * flux, depth);
                            vrl.setEndPoint(mi.p);
                            m_vrlMap->push_back(std::move(vrl));
                        }
                    }
                    // ------------------ Phase function sampling -----------------
                    masked(phase, !act_medium_scatter) = nullptr;
                    auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                    RayDifferential3f new_ray(mi.spawn_ray(wo));
                    new_ray.mint                    = 0.0f;
                    masked(ray, act_medium_scatter) = new_ray;
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

                    handleSurfaceInteraction(ray, depth, delta, si, medium, flux * throughput);

                    BSDFContext bCtx;
                    BSDFPtr bsdf = si.bsdf(ray);

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
                    delta = non_null_bsdf && (has_flag(bs.sampled_type, BSDFFlags::Transmission) || has_flag(bs.sampled_type, BSDFFlags::Reflection));

                    /*if (any_or<true>(has_flag(bs.sampled_type, BSDFFlags::Transmission)))
                        Log(LogLevel::Info, "Delta interact");*/

                    Mask intersect2             = active_surface && needs_intersection;
                    SurfaceInteraction3f si_new = si;
                    if (any_or<true>(intersect2))
                        si_new = scene->ray_intersect(ray, active);
                    needs_intersection &= !intersect2;

                    Mask has_medium_trans            = active_surface && si.is_medium_transition();
                    masked(medium, has_medium_trans) = si.target_medium(ray.d);

                    si = si_new;
                }

                if (depth >= greatestDepth)
                    greatestDepth = depth;

                active &= (active_surface | active_medium);
            }
        }

        std::string desad = "greatest depth = " + std::to_string(greatestDepth);
        Log(LogLevel::Info, desad.c_str());

        float scale          = 1.0 / m_numLightEmissions;
        std::string debugStr = "Global Photon scale: " + std::to_string(scale);
        Log(LogLevel::Info, debugStr.c_str());
        debugStr = "Num Light Emissions: " + std::to_string(m_numLightEmissions);
        Log(LogLevel::Info, debugStr.c_str());

        Log(LogLevel::Info, "Depth Counter: ");
        for (size_t i = 0; i < depthCounter.size(); i++) {
            debugStr = "- d" + std::to_string(i + 1) + " = " + std::to_string(depthCounter[i]);
            Log(LogLevel::Info, debugStr.c_str());
        }

        m_globalPhotonMap->setScaleFactor(scale);
        m_globalPhotonMap->build();
        debugStr = "Building global PM, size: " + std::to_string(m_globalPhotonMap->size());
        Log(LogLevel::Info, debugStr.c_str());

        if (m_causticPhotonMap->size() > 0) {
            m_causticPhotonMap->setScaleFactor(scale);
            debugStr = "Building caustic PM, size: " + std::to_string(m_causticPhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            m_causticPhotonMap->build();
        } else {
            Log(LogLevel::Info, "No caustic photons");
        }
        
        if (m_vrlMap->size() > 0)
        {    
            // If needed, we can change the VRL
            if (m_longVRL) {
                Log(LogLevel::Info, "Transform short VRL into long...");
                m_vrlMap->toLong(scene);
            }
            if (m_diceVRL > 1) {
                Log(LogLevel::Info, "Dicing the VRL...");
                m_vrlMap->dicingVRL(scene, sampler, m_diceVRL);
            }

            debugStr = "Building  VRL Map, size: " + std::to_string(m_vrlMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            // If needed, an acceleration data structure is build on the fly

            // is this correct?
            m_vrlMap->setScaleFactor(1.0 / m_vrlMap->size());
            m_vrlMap->build(scene, m_useLightCut ? ELightCutAcceleration : ENoVRLAcceleration, sampler, m_thresholdBetterDist, m_thresholdError);
        } else {
            Log(LogLevel::Info, "No VRLs");
        }
        
        Log(LogLevel::Info, "Pre Processing done.");
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &_ray, const Medium *_medium, Float *aovs, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static Float m_globalLookupRadius = -1, m_causticLookupRadius = -1;
        if (m_globalLookupRadius == -1) {
            Float sceneRadius        = norm(scene->bbox().center() - scene->bbox().max);
            m_globalLookupRadius     = m_globalLookupRadiusRelative * sceneRadius;
            m_causticLookupRadius    = m_causticLookupRadiusRelative * sceneRadius;
            std::string lookupString = "- Global Lookup Radius: " + std::to_string(m_globalLookupRadius);
            Log(LogLevel::Info, lookupString.c_str());
            lookupString = "- Scene Radius: " + std::to_string(sceneRadius);
            Log(LogLevel::Info, lookupString.c_str());
        }

        Ray3f ray(_ray);
        MediumPtr medium = _medium;
        Spectrum radiance(0.0f), throughput(1.0f);

        MediumInteraction3f mi  = zero<MediumInteraction3f>();
        mi.t                    = math::Infinity<Float>;
        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = math::Infinity<Float>;

        si             = scene->ray_intersect(ray, active);
        Mask valid_ray = si.is_valid();

        float eta(1.0f);
        int nullInteractions = 0;
        bool delta = false, lastNullInteraction = false;
        int photonEstimaterCount = 0;

        // medium = si.target_medium(ray.d);
        Mask needs_intersection = true;
        UInt32 depth            = 0;

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        for (int bounce = 0;; ++bounce) {
            sampler->advance();

            active &= any(neq(depolarize(throughput), 0.f));
            Float q         = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rrDepth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            masked(throughput, perform_rr) *= rcp(detach(q));

            Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
            if (none(active) || all(exceeded_max_depth))
                break;

            // -------------------- RTE ----------------- //

            Mask active_medium    = active && neq(medium, nullptr);
            Mask active_surface   = active && !active_medium;
            Mask act_null_scatter = false, act_medium_scatter = false, escaped_medium = false;

#pragma region RTE
            Mask is_spectral  = active_medium;
            Mask not_spectral = false;
            if (any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (any_or<true>(active_medium)) {
                mi                                                                           = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                Mask intersect                                                               = needs_intersection && active_medium;
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

                if (any_or<true>(act_medium_scatter)) {
                    Float totalLength = norm(mi.p - ray.o);
                    Ray3f cameraRay(ray);
                    cameraRay.maxt = mi.t;
                    auto [evaluations, color, intersections] =
                        m_vrlMap->query(cameraRay, scene, sampler, -1, totalLength, m_useUniformSampling, m_RRVRL ? EDistanceRoulette : ENoRussianRoulette, m_scaleRR, channel);
                    masked(radiance, active) += color * throughput;

                    if (medium->is_homogeneous())
                        break;
                }
                // masked(radiance, active) += m_volumePhotonMap->estimateRadianceVolume(si, mi, m_globalLookupRadius, m_globalLookupSize) * throughput;
                // ------------------ Phase function sampling -----------------
                masked(phase, !act_medium_scatter) = nullptr;
                auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                Ray3f new_ray                      = mi.spawn_ray(wo);
                new_ray.mint                       = 0.0f;
                masked(ray, act_medium_scatter)    = new_ray;
                needs_intersection |= act_medium_scatter;
            }

#pragma endregion

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);

            active_surface &= si.is_valid();

            // -------------------- End RTE ----------------- //

            if (any_or<true>(active_surface)) {

                if (si.shape->is_emitter()) {
                    Spectrum emitterEval = si.shape->emitter()->eval(si);
                    radiance += emitterEval * throughput;
                    break;
                }

                BSDFContext bCtx;
                BSDFPtr bsdf = si.bsdf(ray);

                auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                bsdfVal            = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && !has_flag(bsdf->flags(), BSDFFlags::Transmission);
                // && !has_flag(bs.sampled_type, BSDFFlags::Reflection);
                // Photon Map Sampling
                if (likely(any_or<true>(active_e))) {
                    if (photonEstimaterCount == 0) {
                        // direct method without final gather.
                        radiance[active_surface] += m_causticPhotonMap->estimateCausticRadiance(si, m_causticLookupRadius, m_causticLookupSize) * throughput;
                        ++photonEstimaterCount;
                        radiance[active_surface] += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize) * throughput;
                        break;

                    } else if (photonEstimaterCount == 1) {
                    }
                }

                masked(throughput, active_surface) *= bsdfVal;
                masked(eta, active_surface) *= bs.eta;

                Ray3f bsdf_ray(si.spawn_ray(si.to_world(bs.wo)));
                masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                masked(depth, non_null_bsdf) += 1;
                masked(nullInteractions, !non_null_bsdf) += 1;

                valid_ray |= non_null_bsdf;
                delta = non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);

                Mask intersect2             = active_surface && needs_intersection;
                SurfaceInteraction3f si_new = si;
                if (any_or<true>(intersect2))
                    si_new = scene->ray_intersect(ray, active);
                needs_intersection &= !intersect2;

                Mask has_medium_trans            = active_surface && si.is_medium_transition();
                masked(medium, has_medium_trans) = si.target_medium(ray.d);

                masked(si, intersect2) = si_new;
            }
            active &= (active_surface | active_medium);
        }
        // Photon map is used for both direct & indirect radiance.
        radiance *= 2;
        /* std::ostringstream stream;
         stream << "Final radiance: " << radiance;
         std::string str = stream.str();
         Log(LogLevel::Info, str.c_str());*/
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
                UInt32 index            = min(UInt32(sample.x() * (ScalarFloat) scene->emitters().size()), (uint32_t) scene->emitters().size() - 1);
                sample.x()              = (sample.x() - index * emitter_pdf) * scene->emitters().size();
                emitter                 = scene->emitters()[index];
            }
        }

        return emitter;
    }

    void handleSurfaceInteraction(const Ray3f &ray, int depth, bool delta, const SurfaceInteraction3f &si, const Medium *medium, const Spectrum &weight) const {
        BSDFPtr bsdf       = si.bsdf();
        uint32_t bsdfFlags = bsdf->flags();
        PhotonData photonData(si.n, ray.d, weight, depth);
        if (!has_flag(bsdf->flags(), BSDFFlags::Smooth))
            return;
        if (!delta) {
            m_globalPhotonMap->insert(si.p, std::move(photonData));
        } else {
            m_causticPhotonMap->insert(si.p, std::move(photonData));
        }
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
    int m_numLightEmissions, m_maxDepth;

    int m_nbParticules;
    bool m_useLightCut;

    // Options for VRL
    bool m_longVRL;
    int m_diceVRL;
    bool m_useUniformSampling;
    bool m_RRVRL;
    Float m_scaleRR;
    int m_nbSamplesRay; //< Mostly for the VPL
    bool m_shootCenter;
    int m_rrDepth;

    // Option for LC (VRL)
    int m_thresholdBetterDist;

    // Option for LC
    float m_thresholdError;

    // Internal
    VRLMap* m_vrlMap;

    // For counting the number of iteration
    std::vector<Float> percentage_nbVRLEval;
    std::vector<Float> percentage_nbBBIntersection;

    // ***************** Surface Illumination ***************** //
    PhotonMap *m_globalPhotonMap;
    PhotonMap *m_causticPhotonMap;

    int m_directSamples, m_glossySamples, m_maxSpecularDepth, m_granularity;
    int m_minDepth = 1;
    int m_globalPhotons, m_causticPhotons, m_targetVRLs;
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

MTS_IMPLEMENT_CLASS_VARIANT(VRLIntegrator, SamplingIntegrator);
MTS_EXPORT_PLUGIN(VRLIntegrator, "VRL integrator");
NAMESPACE_END(mitsuba)
