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
const int numPhotons = 30000;

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
        m_globalLookupRadiusRelative = props.float_("globalLookupRadiusRelative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize    = props.int_("globalLookupSize", 120);
        m_causticLookupSize   = props.int_("causticLookupSize", 120);
        m_volumeLookupSize    = props.int_("volumeLookupSize", 120);
        m_gatherLocally       = props.bool_("gatherLocally", true);
        m_autoCancelGathering = props.bool_("autoCancelGathering", true);       
    }

    void preprocess(Scene* scene, Sensor* sensor) override {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        Sampler *sampler = sensor->sampler();
        sampler->seed(0);

        std::vector<int> depthCounter;


        // ------------------- Debug Info -------------------------- //
        std::string numPhotonsStr = "- Photon Count: " + std::to_string(numPhotons);
        Log(LogLevel::Info, numPhotonsStr.c_str());
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string photonString = "- Emitter Count: " + std::to_string(emitters.size());
        Log(LogLevel::Info, photonString.c_str());

        for (int i = 0; i < emitters.size(); i++) {
            std::string emitterType = "- E" + std::to_string(i) + " = " + typeid(&emitters[i]).name();
            Log(LogLevel::Info, emitterType.c_str());
        }

        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

        int numShot = 0;
        static int greatestDepth = 0;

        for (int index = 0; index < numPhotons; index++) {
            sampler->advance();
            EmitterPtr emitter = nullptr;
            MediumPtr medium = nullptr;
            Spectrum flux(1.0f), throughput(1.0f);
            MediumInteraction3f mi = zero<MediumInteraction3f>();
            mi.t                   = math::Infinity<Float>;
            SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
            si.t                    = math::Infinity<Float>;
            Interaction3f its;

            // Sample random emitter
            auto tuple = sample_emitter_direction(scene, its, sampler->next_2d(), false, true);
            emitter = std::get<2>(tuple);

            auto rayColorPair = emitter->sample_ray(0.0, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            Ray3f ray(rayColorPair.first);
            // TODO: Get flux from emitter here
            // - somehow sample_ray always returns 0.0, while eval returns the correct radiance?
            //flux = rayColorPair.second; 

            float eta(1.0f);
            int nullInteractions = 0;
            bool delta = false, lastNullInteraction = false;
            ++numShot;

            //
            Mask active             = true;
            Mask needs_intersection = true;
            Mask specular_chain     = active && !m_hide_emitters;
            UInt32 depth            = 0;

            UInt32 channel = 0;
            if (is_rgb_v<Spectrum>) {
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
            }

            si = scene->ray_intersect(ray, active);
            
            for (int bounce = 0;; ++bounce) {
                sampler->advance();

                active &= any(neq(depolarize(throughput), 0.f));
                Float q         = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                Mask perform_rr = (depth > (uint32_t) m_rrStartDepth);
                active &= sampler->next_1d(active) < q || !perform_rr;
                masked(throughput, perform_rr) *= rcp(detach(q));

                Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
                if (none(active) || all(exceeded_max_depth))
                    break;

                // -------------------- RTE ----------------- //

                Mask active_medium    = active && neq(medium, nullptr);
                Mask active_surface   = active && !active_medium;
                Mask act_null_scatter = false, act_medium_scatter = false, escaped_medium = false;

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

                    handleMediumInteraction(depth, nullInteractions, delta, mi, medium, -ray.d, flux * throughput);

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

                    // --------------------- Emitter sampling ---------------------
                    /*Mask sample_emitters = mi.medium->use_emitter_sampling();
                    valid_ray |= act_medium_scatter;
                    specular_chain &= !act_medium_scatter;
                    specular_chain |= act_medium_scatter && !sample_emitters;

                    Mask active_e = act_medium_scatter && sample_emitters;
                    if (any_or<true>(active_e)) {
                        auto [emitted, ds] = sample_emitter(mi, true, scene, sampler, medium, channel, active_e);
                        Float phase_val    = phase->eval(phase_ctx, mi, ds.d, active_e);
                        masked(result, active_e) += throughput * phase_val * emitted;
                    }*/

                    // ------------------ Phase function sampling -----------------
                    masked(phase, !act_medium_scatter) = nullptr;
                    auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                    Ray3f new_ray                      = mi.spawn_ray(wo);
                    new_ray.mint                       = 0.0f;
                    masked(ray, act_medium_scatter)    = new_ray;
                    needs_intersection |= act_medium_scatter;
                }

                // --------------------- Surface Interactions ---------------------
                active_surface |= escaped_medium;
                Mask intersect = active_surface && needs_intersection;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);

                active_surface &= si.is_valid();

                // -------------------- End RTE ----------------- //

                if (any_or<true>(active_surface)) {

                    BSDFContext bCtx;
                    BSDFPtr bsdf  = si.bsdf(ray);
                    Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth);

                    handleSurfaceInteraction(depth, delta, si, medium, flux * throughput);

                    auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                    bsdfVal            = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

                    masked(throughput, active_surface) *= bsdfVal;
                    masked(eta, active_surface) *= bs.eta;

                    // For some reason, this way of bouncing causes no white splotches, at the expense of only having 2 bounces max.
                    /*ray.o    = si.p;
                    ray.d    = si.to_world(bs.wo);
                    ray.mint = math::RayEpsilon<Float>;*/
                    // wheras the correct way does, and allows photons to bounce around like they should.
                    Ray3f bsdf_ray                = si.spawn_ray(si.to_world(bs.wo));
                    masked(ray, active_surface) = bsdf_ray;
                    needs_intersection |= active_surface;

                    Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                    masked(depth, non_null_bsdf) += 1;
                    masked(nullInteractions, !non_null_bsdf) += 1;

                    valid_ray |= non_null_bsdf;
                    delta = non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                    specular_chain |= delta;
                    specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));

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
            
        }

        std::string desad = "greatest depth = " + std::to_string(greatestDepth);
        Log(LogLevel::Info, desad.c_str());

        float scale          = 1.0 / m_globalPhotonMap->size();
        std::string debugStr = "Global Photon scale: " + std::to_string(scale);
        Log(LogLevel::Info, debugStr.c_str());
        debugStr = "Num shot: " + std::to_string(numShot);
        Log(LogLevel::Info, debugStr.c_str());

        Log(LogLevel::Info, "Depth Counter: ");
        for (size_t i = 0; i < depthCounter.size(); i++) {
            debugStr = "- d" + std::to_string(i + 1) + " = " + std::to_string(depthCounter[i]);
            Log(LogLevel::Info, debugStr.c_str());
        }

        m_globalPhotonMap->setScaleFactor(1.0 / numShot);
        m_globalPhotonMap->build();

        if (m_causticPhotonMap->size() > 0) {
            m_causticPhotonMap->setScaleFactor(1.0f / numShot);
            debugStr = "Building caustic PM, size: " + std::to_string(m_causticPhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            m_causticPhotonMap->build();
        }

        if (m_volumePhotonMap->size() > 0) {
            m_volumePhotonMap->setScaleFactor(1.0f / numShot);
            debugStr = "Building volume PM, size: " + std::to_string(m_volumePhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
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


        Ray3f ray(_ray);
        Spectrum radiance(0.0f), throughput(1.0f);

        MediumInteraction3f mi  = zero<MediumInteraction3f>();
        mi.t                    = math::Infinity<Float>;
        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = math::Infinity<Float>;
        
        si = scene->ray_intersect(ray, active);
        Mask valid_ray = si.is_valid();

        float eta(1.0f);
        int nullInteractions = 0;
        bool delta = false, lastNullInteraction = false;

        // medium = si.target_medium(ray.d);
        Mask needs_intersection = true;
        Mask specular_chain     = active && !m_hide_emitters;
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
            Mask perform_rr = (depth > (uint32_t) m_rrStartDepth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            masked(throughput, perform_rr) *= rcp(detach(q));

            Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
            if (none(active) || all(exceeded_max_depth))
                break;

            // -------------------- RTE ----------------- //

            Mask active_medium    = active && neq(medium, nullptr);
            Mask active_surface   = active && !active_medium;
            Mask act_null_scatter = false, act_medium_scatter = false, escaped_medium = false;

            Mask is_spectral  = active_medium;
            Mask not_spectral = false;
            if (any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (any_or<true>(active_medium)) {
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

                // --------------------- Emitter sampling ---------------------
                /*Mask sample_emitters = mi.medium->use_emitter_sampling();
                valid_ray |= act_medium_scatter;
                specular_chain &= !act_medium_scatter;
                specular_chain |= act_medium_scatter && !sample_emitters;

                Mask active_e = act_medium_scatter && sample_emitters;
                if (any_or<true>(active_e)) {
                    auto [emitted, ds] = sample_emitter(mi, true, scene, sampler, medium, channel, active_e);
                    Float phase_val    = phase->eval(phase_ctx, mi, ds.d, active_e);
                    masked(result, active_e) += throughput * phase_val * emitted;
                }*/

                masked(radiance, active) += m_bre->query(ray, medium, si, sampler, channel, active, m_maxDepth - 1, false);
                //masked(radiance, active) += m_volumePhotonMap->estimateRadianceVolume(si, mi, m_globalLookupRadius, m_globalLookupSize);

                // ------------------ Phase function sampling -----------------
                masked(phase, !act_medium_scatter) = nullptr;
                auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                Ray3f new_ray                      = mi.spawn_ray(wo);
                new_ray.mint                       = 0.0f;
                masked(ray, act_medium_scatter)    = new_ray;
                needs_intersection |= act_medium_scatter;
            }

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);

            active_surface &= si.is_valid();

            // -------------------- End RTE ----------------- //

            if (any_or<true>(active_surface)) {

                BSDFContext bCtx;
                BSDFPtr bsdf  = si.bsdf(ray);
                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth);

                // Photon Map Sampling
                if (likely(any_or<true>(active_e))) {
                    radiance[active] += m_causticPhotonMap->estimateRadiance(si, m_causticLookupRadius, m_causticLookupSize) * throughput;
                    radiance[active] += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize) * throughput;
                    //break;
                    // for correctness we should bbreak here, indirect lighting or of multiple bounces should be received from photon map
                    // however doing so with multiple bounces in preprocess will yield a few high intensity photons (maybe cause of specular?)
                    // and no indirect lighting.... so for now, just use this as a cheat.
                }

                auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                bsdfVal            = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

                masked(throughput, active_surface) *= bsdfVal;
                masked(eta, active_surface) *= bs.eta;

                // For some reason, this way of bouncing causes no white splotches, at the expense of only having 2 bounces max.
                /*ray.o    = si.p;
                ray.d    = si.to_world(bs.wo);
                ray.mint = math::RayEpsilon<Float>;*/
                // wheras the correct way does, and allows photons to bounce around like they should.
                Ray3f bsdf_ray              = si.spawn_ray(si.to_world(bs.wo));
                masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                masked(depth, non_null_bsdf) += 1;
                masked(nullInteractions, !non_null_bsdf) += 1;

                valid_ray |= non_null_bsdf;
                delta = non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain |= delta;
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));

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

        return { radiance, valid_ray };
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

    void handleSurfaceInteraction(int depth, bool delta,
                                  const SurfaceInteraction3f &si,
                                  const Medium *medium,
                                  const Spectrum &weight) const {
        BSDFPtr bsdf       = si.bsdf();
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
