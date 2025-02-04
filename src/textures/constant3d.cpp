#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class ConstVolume final : public Volume<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Volume, m_world_to_local)
    MTS_IMPORT_TYPES(Texture)

    ConstVolume(const Properties &props) : Base(props) {
        m_color = props.texture<Texture>("color", 1.f);
    }

    UnpolarizedSpectrum eval(const Interaction3f &it, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return eval_impl(it, active);
    }

    Float eval_1(const Interaction3f & /* it */, Mask /* active */) const override {
        return m_color->mean();
    }


    MTS_INLINE auto eval_impl(const Interaction3f &it, const Mask &active) const {
        SurfaceInteraction3f si;
        si.uv          = Point2f(0.f, 0.f);
        si.wavelengths = it.wavelengths;
        si.time        = it.time;
        auto result = m_color->eval(si, active);
        return result;
    }


    ScalarFloat max() const override { return m_color->mean(); }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("color", m_color.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ConstVolume[" << std::endl
            << "  world_to_local = " << m_world_to_local << "," << std::endl
            << "  color = " << m_color << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ref<Texture> m_color;
};

MTS_IMPLEMENT_CLASS_VARIANT(ConstVolume, Volume)
MTS_EXPORT_PLUGIN(ConstVolume, "Constant 3D texture")
NAMESPACE_END(mitsuba)
