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

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class VRLIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium)

    VRLIntegrator(const Properties &props) : Base(props) {
    }
        
    std::pair<Spectrum, Mask>
    sample(const Scene *scene, Sampler * /* sampler */,
           const RayDifferential3f &ray, const Medium * /* medium */,
           Float * /* aovs */, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);

        return { select(si.is_valid(), si.t, 0.f), si.is_valid() };
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(VRLIntegrator, MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(VRLIntegrator, "VRL integrator");
NAMESPACE_END(mitsuba)
