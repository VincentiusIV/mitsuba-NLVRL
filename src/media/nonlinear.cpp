#include <enoki/stl.h>

#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template<typename Float, typename Spectrum, typename AABB> 
struct NLNode {
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    AABB aabb;
    float ior;

    inline NLNode() { 
        aabb = AABB();
        ior = 1.0f;
    }

    inline NLNode(AABB _aabb, float _ior) : aabb(_aabb), ior(_ior) {

    }
};

template <typename Float, typename Spectrum> 
class NonLinearMedia final : public Medium<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction)
    MTS_IMPORT_TYPES(Scene, Sampler, Texture, Volume, Shape)

    typedef NLNode<Float, Spectrum, ScalarBoundingBox3f> NLNode;

    NonLinearMedia(const Properties &props) : Base(props) {
        m_is_homogeneous          = true;
        m_is_nonlinear            = true;
        m_albedo                  = props.volume<Volume>("albedo", 0.75f);
        m_sigmat                  = props.volume<Volume>("sigma_t", 1.f);
        m_scale                   = props.float_("scale", 1.0f);
        m_max_density             = m_scale * m_sigmat->max();
        m_inv_max_density         = 1.0f / (m_scale * m_sigmat->max());
        m_has_spectral_extinction = props.bool_("has_spectral_extinction", true);
        resolution                = Point3f(props.int_("res_x", 2.0), props.int_("res_y", 2.0), props.int_("res_z", 2.0));

    }

    void build(Point3f min, Point3f max) override {
        bbox = ScalarBoundingBox3f(min, max);
        width          = bbox.extents().x();
        height         = bbox.extents().y();
        depth          = bbox.extents().z();
        cellSize       = Point3f(width / resolution[0], height / resolution[1], depth / resolution[2]);

        int arraySize = resolution[0] * resolution[1] * resolution[2];
        int arrayIndex = 0;
        grid           = new NLNode[arraySize];
        Log(LogLevel::Info,"[NLHM]: Allocating grid... size = %i", arraySize);
        Log(LogLevel::Info, to_string().c_str());
        for (int x = 0; x < resolution[0]; x++) {
            for (int y = 0; y < resolution[1]; y++) {
                for (int z = 0; z < resolution[2]; z++) {
                    Point3f min = bbox.min + Point3f(x * cellSize[0], y * cellSize[1], z * cellSize[2]);
                    Point3f max = min + cellSize;
                    ScalarBoundingBox3f newBox(min, max);
                    
                    NLNode newNode(newBox, calculateIoR(newBox.center()));
                    grid[arrayIndex++] = std::move(newNode);
                }
            }
        }   
    }

    float calculateIoR(Point3f position) const {
        // temp
        Point3f relativePosition = position - bbox.min;
        float norm = relativePosition[1] / height;
        return lerp(1.0f, 0.89f, norm);
    }

    NLNode getNode(Point3f position) const { 
        if (!bbox.contains(position))
            return NLNode();
        Point3f relativePosition = position - bbox.min;
        Point3f coordinate = relativePosition / cellSize;
        coordinate = enoki::floor(coordinate);
        int arrayIndex           = (coordinate[2] * resolution[0] * resolution[1]) + (coordinate[1] * resolution[0]) + coordinate[0];
        return grid[arrayIndex];
    }

    Vector3f getNormal(Point3f position) const { 
        Vector3f minDiff = position - bbox.min;
        Vector3f maxDiff = position - bbox.max;

        if (enoki::abs(minDiff[0]) < math::Epsilon<Float>)
            return Vector3f(-1, 0, 0);
        else if (enoki::abs(maxDiff[0]) < math::Epsilon<Float>)
            return Vector3f(1, 0, 0);
        else if (enoki::abs(minDiff[1]) < math::Epsilon<Float>)
            return Vector3f(0, -1, 0);
        else if (enoki::abs(maxDiff[1]) < math::Epsilon<Float>)
            return Vector3f(0, 1, 0);
        else if (enoki::abs(minDiff[2]) < math::Epsilon<Float>)
            return Vector3f(0, 0, -1);
        else if (enoki::abs(maxDiff[2]) < math::Epsilon<Float>)
            return Vector3f(0, 0, 1);
        return Vector3f(0);
    }

    NonLinearInteraction sampleNonLinearInteraction(const Ray3f &ray, UInt32 channel, Mask active) const override { 

        NonLinearInteraction nlits;
        if (!bbox.contains(ray.o))
            return std::move(nlits); 
            
        nlits.is_valid = true;
   
        // 1. Find AABB that ray.o resides in.
        NLNode node = getNode(ray.o);
        nlits.n1 = node.ior;

        // 2. Intersect that AABB with the ray, find maxt intersection point.
        auto [aabb_its, mint, maxt] = node.aabb.ray_intersect(ray);
        aabb_its &= (enoki::isfinite(mint) || enoki::isfinite(maxt));
        active &= aabb_its;
        masked(mint, !active) = 0.f;
        masked(maxt, !active) = math::Infinity<Float>;

        mint    = max(ray.mint, mint);
        maxt    = min(ray.maxt, maxt);

        nlits.t = maxt;
        nlits.p                  = ray(maxt);

        // 3. Calculate normal
        nlits.n = getNormal(nlits.p);
        
        // 4. Find neighbouring node that was hit, fill in n2.
        Point3f neighbourOrigin = nlits.p + ray.d * math::Epsilon<Float>;
        NLNode neighbour        = getNode(neighbourOrigin);
        nlits.n2 = neighbour.ior;

        return nlits; 
    }

    MTS_INLINE auto eval_sigmat(const MediumInteraction3f &mi) const { return m_sigmat->eval(mi) * m_scale; }

    UnpolarizedSpectrum get_combined_extinction(const MediumInteraction3f &mi, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        return eval_sigmat(mi);
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum> get_scattering_coefficients(const MediumInteraction3f &mi, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        auto sigmat                = eval_sigmat(mi);
        auto sigmas                = sigmat * m_albedo->eval(mi, active);
        UnpolarizedSpectrum sigman = 0.f;
        return { sigmas, sigman, sigmat };
    }

    std::tuple<Mask, Float, Float> intersect_aabb(const Ray3f & /* ray */) const override { return { true, 0.f, math::Infinity<Float> }; }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("scale", m_scale);
        callback->put_object("albedo", m_albedo.get());
        callback->put_object("sigma_t", m_sigmat.get());
        Base::traverse(callback);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "NonLinear-HomogeneousMedium[" << std::endl
            << "  albedo  = " << string::indent(m_albedo) << std::endl
            << "  sigma_t = " << string::indent(m_sigmat) << std::endl
            << "  scale   = " << string::indent(m_scale) << std::endl
            << "  min   = " << string::indent(bbox.min) << std::endl
            << "  max   = " << string::indent(bbox.max) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    ref<Volume> m_sigmat, m_albedo;
    NLNode *grid;

    ScalarBoundingBox3f bbox;
    Vector3f resolution;

    Point3f cellSize;
};

MTS_IMPLEMENT_CLASS_VARIANT(NonLinearMedia, Medium)
MTS_EXPORT_PLUGIN(NonLinearMedia, "Nonlinear Homogeneous Medium")
NAMESPACE_END(mitsuba)
