#pragma once

#include <mitsuba/core/object.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER Medium : public Object {
public:
    MTS_IMPORT_TYPES(PhaseFunction, Sampler, Scene, Texture, Shape);

    struct NonLinearInteraction
    {
    public:
        bool is_valid;
        Vector3f wi, wo, n;
        float n1, n2;
        Point3f p;
        float t;
        float eta;

        inline NonLinearInteraction()
        {
            is_valid = false;
            n1 = n2 = 1.0f;
            t = math::Infinity<Float>;
        }
    };

    /// Intersets a ray with the medium's bounding box
    virtual std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const = 0;

    /// Returns the medium's majorant used for delta tracking
    virtual UnpolarizedSpectrum
    get_combined_extinction(const MediumInteraction3f &mi,
                            Mask active = true) const = 0;

    /// Returns the medium coefficients Sigma_s, Sigma_n and Sigma_t evaluated
    /// at a given MediumInteraction mi
    virtual std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum,
                       UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active = true) const = 0;

    /**
     * \brief Sample a free-flight distance in the medium.
     *
     * This function samples a (tentative) free-flight distance according to an
     * exponential transmittance. It is then up to the integrator to then decide
     * whether the MediumInteraction corresponds to a real or null scattering
     * event.
     *
     * \param ray      Ray, along which a distance should be sampled
     * \param sample   A uniformly distributed random sample
     * \param channel  The channel according to which we will sample the
     * free-flight distance. This argument is only used when rendering in RGB
     * modes.
     *
     * \return         This method returns a MediumInteraction.
     *                 The MediumInteraction will always be valid,
     *                 except if the ray missed the Medium's bounding box.
     */
    MediumInteraction3f sample_interaction(const Ray3f &ray, Float sample,
                                           UInt32 channel, Mask active) const;

    /**
     * \brief Compute the transmittance and PDF
     *
     * This function evaluates the transmittance and PDF of sampling a certain
     * free-flight distance The returned PDF takes into account if a medium
     * interaction occured (mi.t <= si.t) or the ray left the medium (mi.t >
     * si.t)
     *
     * The evaluated PDF is spectrally varying. This allows to account for the
     * fact that the free-flight distance sampling distribution can depend on
     * the wavelength.
     *
     * \return   This method returns a pair of (Transmittance, PDF).
     *
     */
    std::pair<UnpolarizedSpectrum, UnpolarizedSpectrum>
    eval_tr_and_pdf(const MediumInteraction3f &mi,
                    const SurfaceInteraction3f &si, Mask active) const;

    static Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m, eq(idx, 1u)) = spec[1];
            masked(m, eq(idx, 2u)) = spec[2];
        } else {
            ENOKI_MARK_USED(idx);
        }
        return m;
    }

    Spectrum homoEvalTransmittance(Float length) const {
        Float val       = enoki::exp(max_density() * -length);
        Spectrum tr(max_density() != 0 ? val : (Float) 1.0f);
        return tr;
    }

    Spectrum evalTransmittance(const Ray3f &_ray, Sampler *sampler, Mask active, bool analytic = false) const {
        Ray3f ray(_ray);

        if (is_homogeneous()) {
            float negLength = ray.mint - ray.maxt;
            Float val       = enoki::exp(max_density() * negLength);
            Spectrum tr(max_density() != 0 ? val : (Float) 1.0f);
            return tr;
        }

        Float t    = ray.mint;
        Float maxt = ray.maxt;
        MediumInteraction3f mi;
        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel             = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }
        
        SurfaceInteraction3f si;
        si.t = _ray.maxt;

        Spectrum throughput(1.0f);
        while (t < maxt) {
            mi = sample_interaction(ray, sampler->next_1d(), channel, active);
            if (!mi.is_valid())
                break;

            masked(mi.t, active && (si.t < mi.t)) = math::Infinity<Float>;
            auto [tr, free_flight_pdf]            = eval_tr_and_pdf(mi, si, active);
            throughput *= tr; 
           

            t += mi.t;
            si.t -= mi.t;

            Ray3f new_ray = mi.spawn_ray(_ray.d);
            new_ray.maxt  = max(0.0f, maxt - t);
            ray           = std::move(new_ray);
        }

        if (std::isnan(throughput[0])) {
            Log(LogLevel::Error, "huh");
        }

        return throughput;
    }

    virtual NonLinearInteraction sampleNonLinearInteraction(const Ray3f &ray, UInt32 channel, Mask active) const{ 
        return NonLinearInteraction(); }

    virtual bool handleNonLinearInteraction(const Scene *scene, Sampler *sampler, NonLinearInteraction &nli, SurfaceInteraction3f &si, MediumInteraction3f &mi, Ray3f &ray, Spectrum &throughput,
                                            UInt32 channel, Mask active) const {
        return false;
    }

    virtual void build(Point3f min, Point3f max) {
        bbox   = ScalarBoundingBox3f(min, max);
        width  = bbox.extents().x();
        height = bbox.extents().y();
        depth  = bbox.extents().z();
    }

    /// Return the phase function of this medium
    MTS_INLINE const PhaseFunction *phase_function() const {
        return m_phase_function.get();
    }

    /// Returns whether this specific medium instance uses emitter sampling
    MTS_INLINE bool use_emitter_sampling() const { return m_sample_emitters; }

    /// Returns whether this medium is homogeneous
    MTS_INLINE bool is_homogeneous() const { return m_is_homogeneous; }

    MTS_INLINE bool is_nonlinear() const { return m_is_nonlinear; }

    /// Returns whether this medium has a spectrally varying extinction
    MTS_INLINE bool has_spectral_extinction() const {
        return m_has_spectral_extinction;
    }

    MTS_INLINE ScalarFloat inv_max_density() const { return m_inv_max_density; }
    MTS_INLINE ScalarFloat max_density() const { return m_max_density; }

    MTS_INLINE ScalarFloat scale() const { return m_scale; }

    /// Return a string identifier
    std::string id() const override { return m_id; }

    /// Return a human-readable representation of the Medium
    std::string to_string() const override = 0;

    ENOKI_PINNED_OPERATOR_NEW(Float)
    MTS_DECLARE_CLASS()
protected:
    Medium();
    Medium(const Properties &props);
    virtual ~Medium();

protected:
    ref<PhaseFunction> m_phase_function;
    bool m_sample_emitters, m_is_homogeneous, m_is_nonlinear, m_has_spectral_extinction;
    ScalarFloat m_inv_max_density;
    ScalarFloat m_max_density;
    ScalarFloat m_scale;
    /// Identifier (if available)
    std::string m_id;
    ScalarBoundingBox3f bbox;

    float width, height, depth;
};

MTS_EXTERN_CLASS_RENDER(Medium)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
//! @{ \name Enoki support for packets of Medium pointers
// -----------------------------------------------------------------------

// Enable usage of array pointers for our types
ENOKI_CALL_SUPPORT_TEMPLATE_BEGIN(mitsuba::Medium)
    ENOKI_CALL_SUPPORT_METHOD(phase_function)
    ENOKI_CALL_SUPPORT_METHOD(use_emitter_sampling)
    ENOKI_CALL_SUPPORT_METHOD(is_homogeneous)
    ENOKI_CALL_SUPPORT_METHOD(has_spectral_extinction)
    ENOKI_CALL_SUPPORT_METHOD(get_combined_extinction)
    ENOKI_CALL_SUPPORT_METHOD(intersect_aabb)
    ENOKI_CALL_SUPPORT_METHOD(sample_interaction)
    ENOKI_CALL_SUPPORT_METHOD(eval_tr_and_pdf)
    ENOKI_CALL_SUPPORT_METHOD(get_scattering_coefficients)
    ENOKI_CALL_SUPPORT_TEMPLATE_END(mitsuba::Medium)

//! @}
// -----------------------------------------------------------------------
