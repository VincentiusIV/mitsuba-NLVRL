#include <enoki/fwd.h>
#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/records.h>
#include <mitsuba/core/timer.h>
#include <random>

#include "../photonmapper/photonmap.h"
#include "vrl_map.h"
#include "vrl_struct.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class VRLIntegrator : public SamplingIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef VRL<Float, Spectrum> VRL;
    typedef VRLMap<Float, Spectrum> VRLMap;
    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef typename PhotonMap::PhotonData PhotonData;

    VRLIntegrator(const Properties &props) : Base(props) {
        m_numLightEmissions           = props.int_("light_emissions", 1000000);
        m_maxDepth                    = props.int_("max_depth", 512);
        m_directSamples               = props.int_("direct_samples", 16);
        m_glossySamples               = props.int_("glossy_samples", 32);
        m_maxSpecularDepth            = props.int_("max_specular_depth", 4);
        m_granularity                 = props.int_("granularity", 0);
        m_globalPhotons               = props.int_("global_photons", 250000);
        m_causticPhotons              = props.int_("caustic_photons", 250000);
        m_targetVRLs                  = props.int_("target_vrls", 1000);
        m_volumeLookupRadiusRelative  = props.float_("volume_lookup_radius_relative", 0.005f);
        m_globalLookupRadiusRelative  = props.float_("global_lookup_radius_relative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("caustic_lookup_radius_relative", 0.0125f);
        m_globalLookupSize            = props.int_("global_lookup_size", 120);
        m_causticLookupSize           = props.int_("caustic_lookup_size", 120);
        m_samplesPerQuery             = props.int_("samples_per_query", 2);
        m_gatherLocally               = props.bool_("gather_locally", true);
        m_autoCancelGathering         = props.bool_("auto_cancel_gathering", true);
        

        m_nbParticules = props.int_("nb_particules", 50);
        m_rrDepth      = props.int_("rr_depth", 5);

        m_useLaser        = props.bool_("use_laser", false);
        laserOrigin = props.vector3f("laser_origin", Vector3f());
        laserDirection = props.vector3f("laser_direction", Vector3f());

        m_useDirectIllum = props.bool_("use_direct_illum", true);
        // VRL Options
        m_diceVRL             = props.int_("dice_vrl", 1);
        m_longVRL             = props.bool_("long_vrl", false);
        m_useUniformSampling  = props.bool_("use_uniform_sampling", false);


        m_useNonLinear        = props.bool_("use_non_linear", true);
        m_useLightCut = props.bool_("use_light_cut", false);
        m_stochasticLightcut  = props.bool_("stochastic_lightcut", false);
        m_lightcutSamples = props.int_("lightcut_samples", 1);
        m_RRVRL               = props.bool_("rr_vrl", false);
        m_scaleRR             = props.float_("scale_rr", 0.5); // 2 meters before 5%
        m_nbSamplesRay        = props.int_("nb_samples_ray", 1);
        m_thresholdBetterDist = props.int_("threshold_better_dist", 8);
        m_thresholdError      = props.float_("threshold_error", 0.02);
        m_shootCenter         = props.bool_("shoot_center", true);

        m_globalPhotonMap  = new PhotonMap(m_numLightEmissions);
        m_causticPhotonMap = new PhotonMap(m_numLightEmissions);
        Log(LogLevel::Info, "Constructing VRL Map...");
        m_vrlMap = new VRLMap(m_targetVRLs);
    }

    void preprocess(Scene *scene, Sensor *sensor) override {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        m_preprocess_timer.reset();

        for each (auto shape in scene->shapes()) {
            if (shape->interior_medium() != nullptr) {
                ScalarBoundingBox3f shape_bbox = shape->bbox();
                shape->build(shape_bbox.min, shape_bbox.max);
            }
        }

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
        int volumeLightCount = 0;

        for (int index = 0; index < m_numLightEmissions; index++) {
            sampler->advance();
            EmitterPtr emitter = nullptr;
            MediumPtr medium   = nullptr;
            Spectrum throughput(1.0f);
            MediumInteraction3f mi  = zero<MediumInteraction3f>();
            mi.t                    = math::Infinity<Float>;
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
                flux              = rayColorPair.second;

                if (neq(emitter->shape(), nullptr)) {
                    flux = emitter->getUniformRadiance();
                    flux *= math::Pi<float> * emitter->shape()->surface_area();
                }

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
            bool wasTransmitted = false, volumePath = false;
            bool is_direct = true;

            //
            Mask active             = true;
            Mask needs_intersection = true;
            UInt32 depth = 1, channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
            }

            VRL tempVRL(ray.o, medium, throughput * flux, depth, channel, is_direct);

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
                    mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);

                    if (m_useNonLinear && medium->is_nonlinear()) {
                        nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);

                        for (size_t i = 0; i < 100; i++) {

                            if (nli.t > mi.t || !nli.is_valid)
                                break;
                            // check intersection
                            Ray3f its_test(ray);
                            its_test.maxt         = nli.t;
                            si = scene->ray_intersect(its_test, active);
                            if (si.is_valid())
                                break;

                            // Move ray to nli.p + Eps
                            ray.o = ray(nli.t + math::RayEpsilon<Float>);
                            ray.d = nli.wo;
                            ray.update();
                            mi.sh_frame                 = Frame3f(ray.d);
                            mi.wi                       = -ray.d;
                            auto [aabb_its, mint, maxt] = medium->intersect_aabb(ray);
                            aabb_its &= (enoki::isfinite(mint) || enoki::isfinite(maxt));
                            active &= aabb_its;
                            mint = max(ray.mint, mint);
                            maxt = min(ray.maxt, maxt);

                            auto combined_extinction = medium->get_combined_extinction(mi, active_medium);
                            Float m                  = combined_extinction[0];
                            if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
                                masked(m, eq(channel, 1u)) = combined_extinction[1];
                                masked(m, eq(channel, 2u)) = combined_extinction[2];
                            } else {
                                ENOKI_MARK_USED(channel);
                            }

                            Mask valid_mi = active && (mi.t <= maxt);
                            mi.t -= (nli.t + math::RayEpsilon<Float>);
                            mi.p                                         = ray(mi.t);
                            std::tie(mi.sigma_s, mi.sigma_n, mi.sigma_t) = medium->get_scattering_coefficients(mi, valid_mi);
                            mi.combined_extinction                       = combined_extinction;

                            tempVRL.setEndPoint(ray.o);
                            volumePath |= m_vrlMap->push_back(std::move(tempVRL), true);
                            tempVRL = VRL(ray.o, medium, throughput * flux, depth, channel, is_direct);

                            Medium::NonLinearInteraction new_nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);
                            ;
                            nli = std::move(new_nli);
                        }
                    }

                    masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                    Mask intersect = needs_intersection && active_medium;
                    if (any_or<true>(intersect))
                        masked(si, intersect) = scene->ray_intersect(ray, intersect);
                    needs_intersection &= !active_medium;

                    // TODO: Non linear bug -> mi.t is reduced by nli, so transmittance here is now based on the last leftover which result in lower Tr than it should be.

                    Spectrum vrlFlux = throughput * flux;

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

                    if (act_medium_scatter)
                    {
                        tempVRL.setEndPoint(mi.p);
                        volumePath |= m_vrlMap->push_back(std::move(tempVRL), false);
                        if (volumePath)
                            is_direct = false;
                        tempVRL = VRL(mi.p, medium, vrlFlux, depth, channel, is_direct);
                    }

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
                    // ------------------ Phase function sampling -----------------
                    masked(phase, !act_medium_scatter) = nullptr;
                    auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                    Ray3f new_ray(mi.spawn_ray(wo));
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

                    tempVRL.setEndPoint(si.p);
                    volumePath |= m_vrlMap->push_back(std::move(tempVRL), false);

                    handleSurfaceInteraction(ray, depth, wasTransmitted, si, medium, flux * throughput);

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

                    Ray3f bsdf_ray(si.spawn_ray(si.to_world(bs.wo)));

                    masked(ray, active_surface) = bsdf_ray;
                    needs_intersection |= active_surface;

                    Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                    masked(depth, non_null_bsdf) += 1;
                    masked(nullInteractions, !non_null_bsdf) += 1;

                    valid_ray |= non_null_bsdf;
                    wasTransmitted = non_null_bsdf && (has_flag(bs.sampled_type, BSDFFlags::Transmission));

                    if (volumePath && !wasTransmitted)
                        is_direct = false;

                    Mask intersect2             = active_surface && needs_intersection;
                    SurfaceInteraction3f si_new = si;
                    if (any_or<true>(intersect2))
                        si_new = scene->ray_intersect(ray, active);
                    needs_intersection &= !intersect2;

                    Mask has_medium_trans            = active_surface && si.is_medium_transition();

                    bool hadMedium = neq(medium, nullptr);

                    masked(medium, has_medium_trans) = si.target_medium(ray.d);

                    tempVRL = VRL(ray.o, medium, throughput * flux, depth, channel, is_direct);

                    si = si_new;
                }

                if (depth >= greatestDepth)
                    greatestDepth = depth;

                active &= (active_surface | active_medium);
            }

            if (volumePath)
                ++volumeLightCount;           
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

        if (m_vrlMap->size() > 0) {
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
            m_vrlMap->setScaleFactor(1.0f / volumeLightCount);
            m_vrlMap->build(scene, m_useLightCut ? ELightCutAcceleration : ENoVRLAcceleration, sampler, m_thresholdBetterDist, m_thresholdError, m_useUniformSampling, m_useDirectIllum,
                            m_stochasticLightcut, m_lightcutSamples);
        } else {
            Log(LogLevel::Info, "No VRLs");
        }

        Log(Info, "Pre-process finished. (took %s)", util::time_string(m_preprocess_timer.value(), true));
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &_ray, const Medium *_medium, Float *aovs, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static Float m_globalLookupRadius = -1, m_causticLookupRadius = -1, m_volumeLookupRadius = -1;
        if (m_globalLookupRadius == -1) {
            Float sceneRadius        = norm(scene->bbox().center() - scene->bbox().max);
            m_globalLookupRadius     = m_globalLookupRadiusRelative * sceneRadius;
            m_causticLookupRadius    = m_causticLookupRadiusRelative * sceneRadius;
            m_volumeLookupRadius     = m_volumeLookupRadiusRelative * sceneRadius;
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
        mi.t                    = math::Infinity<Float>;
        Medium::NonLinearInteraction nlmi;

        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

        float eta(1.0f);
        int nullInteractions = 0;
        bool delta           = false;

        Mask specular_chain     = active && !m_hide_emitters;
        Mask needs_intersection = true;
        UInt32 depth = 1, channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        for (int bounce = 0;; ++bounce) {
            
            active &= any(neq(depolarize(throughput), 0.f));

            Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
            if (none(active) || all(exceeded_max_depth)) {
                break;
            }

            // -------------------- RTE ----------------- //

            Mask active_medium  = active && neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
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
                Float totalLength = si.t;
                Float t           = 0;

                Ray3f gatherRay(ray.o, ray.d, 0.0f);
                gatherRay.mint = 0.0f;

                if (si.is_valid()) {
                    valid_ray |= true;
                    while (t < si.t) {
                        float length = min(si.t - t, totalLength);
                        gatherRay.maxt = length;
                        auto [evaluations, color, intersections] = m_vrlMap->query(gatherRay, scene, sampler, -1, length, m_useUniformSampling, m_useDirectIllum, m_volumeLookupRadius, m_RRVRL ? EDistanceRoulette : ENoRussianRoulette, m_scaleRR, m_samplesPerQuery, channel);
                        

                        t += length;
                        gatherRay.o = ray(t);

                        mi.t                       = length;
                        mi.p                       = gatherRay.o;
                        mi.wi = -gatherRay.d;

                        auto combined_extinction = medium->get_combined_extinction(mi, active_medium);
                        Float m                  = combined_extinction[0];
                        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
                            masked(m, eq(channel, 1u)) = combined_extinction[1];
                            masked(m, eq(channel, 2u)) = combined_extinction[2];
                        } else {
                            ENOKI_MARK_USED(channel);
                        }
                        mi.combined_extinction = combined_extinction;

                        auto [tr, free_flight_pdf] = medium->eval_tr_and_pdf(mi, si, active);
                        Float tr_pdf               = index_spectrum(free_flight_pdf, channel);
                        //color *= select(tr_pdf > 0, tr / tr_pdf, 0.0f);
                        color *= throughput;
                        radiance += color;
                        throughput *= tr;
                    }
                }

                escaped_medium = true;
                needs_intersection = true;
                medium = nullptr;
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

                // sample global/caustic map

                BSDFContext bCtx;
                BSDFPtr bsdf = si.bsdf(ray);

                auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                bsdfVal            = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && !has_flag(bsdf->flags(), BSDFFlags::Transmission);

                if (likely(any_or<true>(active_e))) {
                    radiance[active_surface] += m_causticPhotonMap->estimateCausticRadiance(si, m_causticLookupRadius, m_causticLookupSize) * throughput;
                    radiance[active_surface] += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize) * throughput;
                    break;
                }

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

    int m_samplesPerQuery = 2;
    int m_nbParticules;
    bool m_useLightCut;
    bool m_useDirectIllum;

    // Options for VRL
    bool m_longVRL;
    int m_diceVRL;
    bool m_useUniformSampling;
    bool m_stochasticLightcut;
    int m_lightcutSamples;
    bool m_useNonLinear;
    bool m_useLaser;
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
    VRLMap *m_vrlMap;

    // For counting the number of iteration
    std::vector<Float> percentage_nbVRLEval;
    std::vector<Float> percentage_nbBBIntersection;

    // ***************** Surface Illumination ***************** //
    PhotonMap *m_globalPhotonMap;
    PhotonMap *m_causticPhotonMap;

    Vector3f laserOrigin, laserDirection;

    int m_directSamples, m_glossySamples, m_maxSpecularDepth, m_granularity;
    int m_minDepth = 1;
    int m_globalPhotons, m_causticPhotons, m_targetVRLs;
    float m_globalLookupRadiusRelative, m_causticLookupRadiusRelative, m_volumeLookupRadiusRelative;
    float m_invEmitterSamples, m_invGlossySamples;
    int m_globalLookupSize, m_causticLookupSize, m_volumeLookupSize;
    /* Should photon gathering steps exclusively run on the local machine? */
    bool m_gatherLocally;
    /* Indicates if the gathering steps should be canceled if not enough photons
     * are generated. */
    bool m_autoCancelGathering;

    
    Timer m_preprocess_timer;
};

MTS_IMPLEMENT_CLASS_VARIANT(VRLIntegrator, SamplingIntegrator);
MTS_EXPORT_PLUGIN(VRLIntegrator, "VRL integrator");
NAMESPACE_END(mitsuba)