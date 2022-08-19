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
    typedef NLRay<Float, Spectrum> NLRay;
    typedef VRLMap<Float, Spectrum> VRLMap;
    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef typename PhotonMap::PhotonData PhotonData;

    VRLIntegrator(const Properties &props) : Base(props) {
        m_globalPhotons               = props.int_("global_photons", 1000);
        m_causticPhotons = props.int_("caustic_photons", 1000);
        m_volumePhotons  = props.int_("volume_photons", 1000);

        m_maxDepth                    = props.int_("max_depth", 512);
        
        m_globalPhotons               = props.int_("global_photons", 250000);
        m_causticPhotons              = props.int_("caustic_photons", 250000);
        m_targetVRLs                  = props.int_("target_vrls", 1000);
        m_volumeLookupRadiusRelative  = props.float_("volume_lookup_radius_relative", 0.005f);
        m_globalLookupRadiusRelative  = props.float_("global_lookup_radius_relative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("caustic_lookup_radius_relative", 0.0125f);
        m_globalLookupSize            = props.int_("global_lookup_size", 120);
        m_causticLookupSize           = props.int_("caustic_lookup_size", 120);
        m_samplesPerQuery             = props.int_("samples_per_query", 2);
        
        m_rrDepth      = props.int_("rr_depth", 5);

        m_useLaser        = props.bool_("use_laser", false);
        laserOrigin = props.vector3f("laser_origin", Vector3f());
        laserDirection           = props.vector3f("laser_direction", Vector3f());
        m_useNonLinear           = props.bool_("use_non_linear", true);
        m_useNonLinearCameraRays = props.bool_("use_non_linear_camera", true);
        m_useNLAtomicQuery  = props.bool_("use_nl_atomic_query", false);

        m_useDirectIllum = props.bool_("use_direct_illum", true);
        // VRL Options
        m_diceVRL             = props.int_("dice_vrl", 1);
        m_longVRL             = props.bool_("long_vrl", false);
        m_useUniformSampling  = props.bool_("use_uniform_sampling", false) || m_useNLAtomicQuery && m_useNonLinearCameraRays;
        Log(Info, "Use NL Camera: %i, NL Atomic: %i, Uniform Sampling: %i", m_useNonLinearCameraRays, m_useNLAtomicQuery, m_useUniformSampling);
        minVRLLength      = props.float_("min_vrl_length", 5);

        m_useLightCut = props.bool_("use_light_cut", false);
        m_stochasticLightcut  = props.bool_("stochastic_lightcut", false);
        m_lightcutSamples = props.int_("lightcut_samples", 1);
        m_RRVRL               = props.bool_("rr_vrl", false);
        m_scaleRR             = props.float_("scale_rr", 0.5); // 2 meters before 5%
        m_thresholdBetterDist = props.int_("threshold_better_dist", 8);
        m_thresholdError      = props.float_("threshold_error", 0.02);
        m_shootCenter         = props.bool_("shoot_center", true);

        Log(LogLevel::Info, "Constructing VRL Map...");
        m_vrlMap           = new VRLMap(m_targetVRLs, m_stochasticLightcut);
        m_globalPhotonMap  = new PhotonMap(m_globalPhotons);
        m_causticPhotonMap = new PhotonMap(m_causticPhotons);
        m_volumePhotonMap  = new PhotonMap(m_useDirectIllum ? m_volumePhotons : 0);

        minNLIcount = math::Max<int>;
        maxNLIcount = math::Min<int>;
    }

    void preprocess(Scene *scene, Sensor *sensor) override {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        m_preprocessTimer.reset();

        for each (auto shape in scene->shapes()) {
            if (shape->interior_medium() != nullptr) {
                ScalarBoundingBox3f shape_bbox = shape->bbox();
                shape->build(shape_bbox.min, shape_bbox.max);
            }
        }

        ref<Sampler> sampler = sensor->sampler()->clone();
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

        static int greatestDepth = 0;
        ScalarFloat emitter_pdf  = 1.f / scene->emitters().size();

        int surfacePathCount = 0, volumePathCount = 0, directVolumePathCount = 0;
        int count = 0;
        while (!m_globalPhotonMap->is_full() || !m_vrlMap->is_full() || !m_volumePhotonMap->is_full()) {
            if (count++ % 100000 == 0) {
                std::ostringstream oss;
                oss << "photons[global=" << string::indent(m_globalPhotonMap->size()) << ", caustic= " << m_causticPhotonMap->size() << ", volume=" << m_volumePhotonMap->size()
                    << ", vrls=" << m_vrlMap->size() << "]";
                Log(LogLevel::Info, oss.str().c_str());
            }
            bool surfacePath = !m_globalPhotonMap->is_full(), volumePath = !m_vrlMap->is_full(), directVolumePath = !m_volumePhotonMap->is_full();

            if (m_globalPhotonMap->is_full() && (m_vrlMap->size() == 0 && m_volumePhotonMap->size() == 0))
                break; // stop, no volume in this scene.

            if ((m_vrlMap->is_full() && m_volumePhotonMap->is_full()) && m_globalPhotonMap->size() == 0)
                break; // stop no surfaces in this scene.

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
                if (neq(emitter->shape(), nullptr))
                    flux *= (M_PI);
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
                        std::vector<VRL> splitVRLs;
                        while (nli.t < mi.t && nli.is_valid) {
                            Vector3f rayDir = ray.d;
                            bool valid = medium->handleNonLinearInteraction(scene, sampler, nli, si, mi, ray, throughput, channel, active_medium);
                            if (!valid)
                                break;
                            if (rayDir != ray.d)
                            {
                                tempVRL.setEndPoint(ray.o);
                                if (tempVRL.length > minVRLLength)
                                {
                                    volumePath |= m_vrlMap->push_back(std::move(tempVRL), false);
                                    tempVRL = VRL(ray.o, medium, throughput * flux, depth, channel, is_direct);
                                }
                            }

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

                    // mediumDepth is only 0 iff it came from a light source OR was transmitted 
                    if (mediumDepth == 0) {
                        directVolumePath |= handleMediumInteraction(depth - nullInteractions, wasTransmitted, mi.p, medium, -ray.d, flux * throughput);
                    }
                    ++mediumDepth;

                    if (act_medium_scatter) {
                        tempVRL.setEndPoint(mi.p);
                        volumePath |= m_vrlMap->push_back(std::move(tempVRL), false);
                        if (volumePath)
                            is_direct = false;
                        tempVRL = VRL(mi.p, medium, throughput * flux, depth, channel, is_direct);
                    }
                    
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

                    surfacePath |= handleSurfaceInteraction(ray, depth, wasTransmitted, si, medium, flux * throughput);

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
                    if (wasTransmitted)
                        mediumDepth = 0;

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

            if (surfacePath && !m_globalPhotonMap->is_full())
                ++surfacePathCount;
            if (volumePath && !m_vrlMap->is_full())
                ++volumePathCount;
            if (directVolumePath && !m_volumePhotonMap->is_full())
                ++directVolumePathCount;         
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

        debugStr = "Caustic Photon scale: " + std::to_string(scale);
        Log(LogLevel::Info, debugStr.c_str());
        if (m_causticPhotonMap->size() > 0) {
            m_causticPhotonMap->setScaleFactor(scale);
            debugStr = "Building caustic PM, size: " + std::to_string(m_causticPhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            m_causticPhotonMap->build();
        } else {
            Log(LogLevel::Info, "No caustic photons");
        }

        if (m_vrlMap->size() > 0) {
            float vrlScale = 1.0f / volumePathCount;
            debugStr       = "VRL scale: " + std::to_string(vrlScale);
            Log(LogLevel::Info, debugStr.c_str());
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
            m_vrlmap_build_timer.reset();
            m_vrlMap->setScaleFactor(vrlScale);
            m_vrlMap->build(scene, m_useLightCut ? ELightCutAcceleration : ENoVRLAcceleration, sampler, m_thresholdBetterDist, m_thresholdError, m_useUniformSampling, m_useDirectIllum, m_lightcutSamples);
            Log(Info, "VRL Map Created. (took %s)", util::time_string(m_vrlmap_build_timer.value(), true));
        } else {
            Log(LogLevel::Info, "No VRLs");
        }

        if (m_volumePhotonMap->size() > 0) {
            m_volumePhotonMap->setScaleFactor(1.0f / directVolumePathCount);
            debugStr = "Building volume PM, size: " + std::to_string(m_volumePhotonMap->size());
            Log(LogLevel::Info, debugStr.c_str());
            m_volumePhotonMap->build();
            // m_bre->build(m_volumePhotonMap, m_volumeLookupSize);
        } else {
            Log(LogLevel::Info, "No volume photons");
        }

        Log(Info, "Pre-process finished. (took %s)", util::time_string(m_preprocessTimer.value(), true));
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
        Medium::NonLinearInteraction nli;

        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

        float eta(1.0f);
        int nullInteractions = 0, mediumDepth = 0;
        bool delta           = false;

        Mask specular_chain     = active && !m_hide_emitters;
        Mask needs_intersection = true;
        UInt32 depth = 1, channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }
        Mask active_medium, active_surface;
        bool wentThruMedium = false;

        int nliCount = 0;

        for (int bounce = 0;; ++bounce) {
            active &= any(neq(depolarize(throughput), 0.f));

            Mask exceeded_max_depth = depth >= (uint32_t) m_maxDepth;
            if (none(active) || all(exceeded_max_depth)) {
                break;
            }


            // -------------------- RTE ----------------- //

            active_medium       = active && neq(medium, nullptr);
            active_surface      = active && !active_medium;
            Mask escaped_medium = false;

            if (bounce > 100000) {
                std::ostringstream stream;
                stream << "over 100k bounces, that cant be right: " << std::endl
                       << "active: " << active << std::endl
                       << "active_medium: " << active_medium << std::endl
                       << "active_surface: " << active_surface << std::endl
                       << "ray: " << ray << std::endl;
                std::string str = stream.str();
                Log(Warn, str.c_str());
                break;
            }

#pragma region RTE
            Mask is_spectral  = active_medium;
            Mask not_spectral = false;
            if (any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (any_or<true>(active_medium)) {
                wentThruMedium = true;

                if (si.is_valid()) {
                    valid_ray = true;
                    nliCount  = 0;
                    // Gather VPM for direct+caustic
                    Float radius   = m_volumeLookupRadius * enoki::lerp(0.75f, 1.25f, sampler->next_1d());
                    Float stepSize = radius, leftOver = 0.0;
                    Float gather_t = 0;

                    Ray3f mediumRay(ray);
                    mediumRay.mint = 0.0f;
                    mediumRay.maxt = radius;

                    size_t MVol = 0;
                    size_t M    = 0;
                    Spectrum volRadiance(0.0f);

                    Ray3f gatherRay(ray);
                    gatherRay.maxt = si.t;
                    Spectrum directIllum(0.0f), indirectIllum(0.0);

                    NLRay nlray;
                    std::vector<Point3f> gatherPoints;

                    if (!si.is_valid()) {
                        Log(Warn, "si invalid before nli");
                    }

                    if (m_useNonLinearCameraRays && m_useNonLinear && medium->is_nonlinear()) {
                        nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);

                        while (nli.t < si.t && si.is_valid() && nli.is_valid) {

                            bool valid = medium->handleNonLinearInteraction(scene, sampler, nli, si, mi, ray, throughput, channel, active_medium);
                            if (!valid)
                                break;
                            ++nliCount;
                            gatherRay.maxt = nli.t;

                            // Add gather points from origin until nli.t
                            gather_t = 0;
                            while (gather_t <= (nli.t - (stepSize - leftOver))) {
                                gather_t += (stepSize - leftOver);
                                leftOver = 0.0f;
                                gatherPoints.push_back(gatherRay(gather_t));
                                stepSize = radius * 2;
                            }
                            leftOver += nli.t - gather_t;

                            nlray.push_back(std::move(gatherRay));

                            gatherRay      = Ray3f(ray);
                            gatherRay.maxt = si.t;

                            nli = medium->sampleNonLinearInteraction(ray, channel, active_medium);
                        }
                    }

                    int debugCount = 0;
                    // Add gather points from origin until end of (last) gatherRay
                    gather_t = 0;
                    while (gather_t <= (gatherRay.maxt - (stepSize - leftOver))) {
                        if (++debugCount > 10000000) {
                            Log(Warn, "prob endless loop in gather point calc?");
                            break;
                        }

                        gather_t += (stepSize - leftOver);
                        leftOver = 0.0f;
                        gatherPoints.push_back(gatherRay(gather_t));
                        stepSize = radius * 2;
                    }

                    nlray.push_back(std::move(gatherRay));

                    maxNLIcount = max(maxNLIcount, nliCount);
                    minNLIcount = min(minNLIcount, nliCount);

                    // Gather Photons for Direct
                    Point3f lastGatherPoint = nlray.o();
                    Spectrum tr             = throughput;
                    for (size_t i = 0; i < gatherPoints.size(); i++) {
                        Point3f gatherPoint = gatherPoints[i];
                        tr *= medium->homoEvalTransmittance(norm(gatherPoint - lastGatherPoint));
                        Spectrum estimate = m_volumePhotonMap->estimateRadianceVolume(gatherPoint, mediumRay.d, medium, sampler, radius, M);
                        directIllum += tr * estimate;
                        lastGatherPoint = gatherPoint;
                    }
                    directIllum *= m_volumePhotonMap->getScaleFactor();
                    radiance += directIllum;

                    if (m_useNLAtomicQuery) {
                        // Gather VRLs for Indirect
                        auto [evaluations, color, intersections] =
                            m_vrlMap->query(nlray, scene, sampler, -1, m_useUniformSampling, m_RRVRL ? EDistanceRoulette : ENoRussianRoulette, m_scaleRR, m_samplesPerQuery, channel);
                        indirectIllum += color * throughput;
                    } else {
                        NLRay temp;
                        Spectrum vrlThroughput = throughput;
                        for (size_t i = 0; i < nlray.parts.size(); i++) {
                            temp.clear();
                            temp.push_back(nlray.parts[i]);
                            auto [evaluations, color, intersections] =
                                m_vrlMap->query(temp, scene, sampler, -1, m_useUniformSampling, m_RRVRL ? EDistanceRoulette : ENoRussianRoulette, m_scaleRR, m_samplesPerQuery, channel);
                            indirectIllum += color * vrlThroughput;
                            vrlThroughput *= medium->homoEvalTransmittance(nlray.parts[i].maxt);
                        }
                    }

                    throughput *= medium->homoEvalTransmittance(nlray.maxt);

                    radiance += indirectIllum;

                    ++mediumDepth;

                    active_surface |= true;
                }
            }

            active &= depth < (uint32_t)m_maxDepth;

#pragma endregion

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);

            if (any_or<true>(active_surface)) {
                // ---------------- Intersection with emitters ----------------
                EmitterPtr emitter = si.emitter(scene);
                Mask use_emitter_contribution = active_surface && specular_chain && neq(emitter, nullptr);
                if (any_or<true>(use_emitter_contribution))
                    masked(radiance, use_emitter_contribution) += throughput * emitter->eval(si, use_emitter_contribution);
            }

            active_surface &= si.is_valid();

            // -------------------- End RTE ----------------- //

            if (any_or<true>(active_surface)) {
                if (si.shape->is_emitter())
                    break;
                valid_ray = true;

                BSDFContext bCtx;
                BSDFPtr bsdf = si.bsdf(ray);

                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth);

                if (likely(any_or<true>(active_e))) {

                    radiance[active_surface] += m_causticPhotonMap->estimateCausticRadiance(si, m_causticLookupRadius, m_causticLookupSize) * throughput;
                    radiance[active_surface] += m_globalPhotonMap->estimateRadiance(si, m_globalLookupRadius, m_globalLookupSize) * throughput;

                    ++surfaceQueryCount;
                    break;
                }

                auto [bs, bsdfVal] = bsdf->sample(bCtx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                bsdfVal = si.to_world_mueller(bsdfVal, -bs.wo, si.wi);

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

                Mask intersect2 = active_surface && needs_intersection;
                SurfaceInteraction3f si_new = si;
                if (any_or<true>(intersect2))
                    si_new = scene->ray_intersect(ray, active);
                needs_intersection &= !intersect2;

                Mask has_medium_trans = active_surface && si.is_medium_transition();
                masked(medium, has_medium_trans) = si.target_medium(ray.d);

                si = si_new;
            }

            active &= (active_surface | active_medium);
        }

        //valid_ray = true;
        /*if (norm(radiance) == 0) {
            Log(Warn, "black rad. ray: %i, nliCount: %i, tr: %i, wrm: %i, am: %i, as: %i, si: %i", ray, nliCount, throughput, wentThruMedium, active_medium, active_surface, si);
        }
        if (!valid_ray) {
            Log(Warn, "invalid ray. ray: %i, tr: %i, wrm: %i, am: %i, as: %i, si: %i", ray, throughput, wentThruMedium, active_medium, active_surface, si);
        }*/


        return { radiance, valid_ray };
    }

    void postprocess(Scene *scene, Sensor *sensor) override {


        std::ostringstream stream;
        stream << "Surface Query Count: " << surfaceQueryCount << std::endl
               << "Volume Gather Count: " << m_volumePhotonMap->queryCount << std::endl
               << "min NLI count: " << minNLIcount << std::endl
               << "max NLI count: " << maxNLIcount << std::endl
               << "VRL Query Count: " << m_vrlMap->queryCount << std::endl
               << "Global Map Size: " << util::mem_string(m_globalPhotonMap->getSize()) << std::endl
               << "Caustic Map Size: " << util::mem_string(m_causticPhotonMap->getSize()) << std::endl
                << "Volume Map Size: " << util::mem_string(m_volumePhotonMap->getSize()) << std::endl
               << "VRL Map Size: " << util::mem_string(m_vrlMap->getSize()) << std::endl;
        std::string str = stream.str();
        Log(LogLevel::Info, str.c_str());
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

        if (neq(scene->environment(), nullptr))
            return scene->environment();

        return emitter;
    }

     bool handleSurfaceInteraction(const Ray3f &ray, int depth, bool wasTransmitted, const SurfaceInteraction3f &si, const Medium *medium, const Spectrum &weight) const {
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

     bool handleMediumInteraction(int depth, bool delta, const Point3f &p, const Medium *medium, const Vector3f &wi, const Spectrum &weight) const {
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
    int m_maxDepth;

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
    bool m_useNonLinearCameraRays;
    bool m_useNLAtomicQuery;
    bool m_useLaser;
    bool m_RRVRL;
    Float m_scaleRR;
    bool m_shootCenter;
    int m_rrDepth;

    float minVRLLength = 5.0f;
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
    PhotonMap *m_volumePhotonMap; // only used for volume caustic and direct illumination

    Vector3f laserOrigin, laserDirection;

    int m_minDepth = 1;
    int m_globalPhotons, m_causticPhotons, m_volumePhotons, m_targetVRLs;
    float m_globalLookupRadiusRelative, m_causticLookupRadiusRelative, m_volumeLookupRadiusRelative;
    float m_invEmitterSamples, m_invGlossySamples;
    int m_globalLookupSize, m_causticLookupSize, m_volumeLookupSize;

    mutable std::atomic<int> surfaceQueryCount, volumeQueryCount, gatherCount, minNLIcount, maxNLIcount;
    mutable std::atomic<size_t> surfaceQueryTime, volumeQueryTime;
    
    Timer m_preprocessTimer, m_vrlmap_build_timer;
};

MTS_IMPLEMENT_CLASS_VARIANT(VRLIntegrator, SamplingIntegrator);
MTS_EXPORT_PLUGIN(VRLIntegrator, "VRL integrator");
NAMESPACE_END(mitsuba)