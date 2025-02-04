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
        topIoR                    = props.float_("top_ior", 0.7f);
        bottomIoR                 = props.float_("bottom_ior", 1.0f);
        iorMethod                 = props.int_("ior_method", 0);
        topTemperature            = props.vector3f("top_temp", Vector3f(15, 0.02, 0));
        bottomTemperature         = props.vector3f("bottom_temp", Vector3f(45, 0, 0));
        topPressure               = props.vector3f("top_pressure", Vector3f(100, 20, 0));
        bottomPressure = props.vector3f("bottom_pressure", Vector3f(1000, 0, 0));
        m_max_density             = m_scale * m_sigmat->max();
        m_inv_max_density         = 1.0f / (m_scale * m_sigmat->max());
        m_has_spectral_extinction = props.bool_("has_spectral_extinction", true);
        m_from_bottom = props.bool_("from_bottom", true);
        resolution                = Point3f(props.int_("res_x", 4.0), props.int_("res_y", 4.0), props.int_("res_z", 4.0));
        

    }

    void build(Point3f min, Point3f max) override {
        Medium::build(min, max);
       
        cellSize       = Point3f(width / resolution[0], height / resolution[1], depth / resolution[2]);

        arraySize = resolution[0] * resolution[1] * resolution[2];
        int arrayIndex = 0;
        grid           = new NLNode[arraySize];
        Log(Info,"[NLHM]: Allocating grid... size = %i", arraySize);
        Float nmin = 100000000, nmax = -10000000;

        for (int x = 0; x < resolution[0]; x++) {
            for (int y = 0; y < resolution[1]; y++) {
                for (int z = 0; z < resolution[2]; z++) {
                    Point3f min = bbox.min + Point3f(x * cellSize[0], y * cellSize[1], z * cellSize[2]);
                    Point3f max = min + cellSize;
                    ScalarBoundingBox3f newBox(min, max);          
                    Float ior          = calculateIoR(newBox.center());
                    NLNode newNode(newBox, ior);
                    nmin = enoki::min(nmin, ior);
                    nmax = enoki::max(nmax, ior);

                    grid[arrayIndex++] = std::move(newNode);
                }
            }
        }
        Log(Info, "[NLHM]: nmin=%i, nmax=%i", nmin, nmax);
    }

    Float P(Float h) const {
        if (h < bottomPressure.y())
            return bottomPressure.x();
        else if (h >= bottomPressure.y() && h < topPressure.y()) {
            Float norm = (h - bottomPressure.y()) / (topPressure.y() - bottomPressure.y());
            return lerp(bottomPressure.x(), topPressure.x(), norm);
        } else {
            return topPressure.y();
        }
    }

    Float T(Float h) const {
        if (h < bottomTemperature.y())
            return bottomTemperature.x();
        else if (h >= bottomTemperature.y() && h < topTemperature.y()) {
            Float norm = (h - bottomTemperature.y()) / (topTemperature.y() - bottomTemperature.y());
            return lerp(bottomTemperature.x(), topTemperature.x(), norm);
        } else {
            return topTemperature.y();
        }
    }

    Float p(Float h) const {
        const Float M = 28.93 * 1e-3; // kg/mol
        const Float R = 8.3145;       // J/mol�K
        return P(h) * M / R * T(h);
    }

    Float n(Float h, Float lambda) const { 
        auto n_l = [](Float lambda) -> Float {
            const float a = 28.79 * 1e-5;
            const float b = 5.67 * 1e-5;
            return a * (1 + b / sqr(lambda)) + 1;
        };

        return p(h) * (n_l(lambda) - 1) + 1;
    }

    float calculateIoR(Point3f position) const {
        // temp
        Point3f relativePosition = position - bbox.min;
        Float distance           = norm(position);
        if (iorMethod == 1) {

            return n(select(m_from_bottom, relativePosition.y(), position.y()), 3);
        } else if (iorMethod == 2) {
            float topY    = topTemperature.y();
            float bottomY = bottomTemperature.y();
            if (position.y() <= bottomY)
                return bottomIoR;
            else if (position.y() >= topY)
                return topY;
            else {
                float norm = (position.y() - bottomY) / (topY - bottomY);
                return lerp(bottomY, topIoR, norm);
            }
        }
        else {
            
            float norm = select(m_from_bottom, relativePosition.y() / height, max(0, position.y() / (height * 0.5))); // 
            return lerp(bottomIoR, topIoR, norm);
        }
    }

    std::pair<Mask, int> getNode(Point3f position) const { 
        if (!bbox.contains(position))
            return { false, 0 };

        for (int i = 0; i < arraySize; i++) {
            if (grid[i].aabb.contains(position))
                return { true, i };
        }

        return { false, 0 };
    }

    std::pair<Mask, int> getNeighbour(int nodeIdx, Vector3f normal) const {
        int nbIdx  = nodeIdx;
        int before = nbIdx;
        assert(nbIdx == before);

        int nz = int(std::floor(normal.z())), ny = int(std::floor(normal.y())), nx = int(std::floor(normal.x()));
        if (abs(nz) == 1.0 && int(resolution.z()) == 1)
            return { false, -1 };

        nbIdx -= nz;

        if (abs(ny) == 1.0 && int(resolution.y()) == 1)
            return { false, -1 };

        nbIdx -= int(ny * resolution[2]);

        if (abs(nx) == 1.0 && int(resolution.x()) == 1)
            return { false, -1 };

        nbIdx -= int(nx * resolution[2] * resolution[1]);

        return { nbIdx >= 0 && nbIdx < arraySize, nbIdx };
    }

    Vector3f getNormal(Point3f position, ScalarBoundingBox3f aabb) const { 
        Vector3f minDiff = enoki::abs(position - aabb.min);
        Vector3f maxDiff = enoki::abs(position - aabb.max);

        Float minDiffExt = hmin(minDiff);
        Float maxDiffExt = hmin(maxDiff);

        Float eps = 1e-03;
        if (minDiffExt < maxDiffExt)
        {
            if (minDiff[0] <= eps && minDiff[0] == minDiffExt)
                return Vector3f(-1.0, 0.0f, 0.0f);
            if (minDiff[1] <= eps && minDiff[1] == minDiffExt)
                return Vector3f(0.0f, -1.0f, 0.0f);
            if (minDiff[2] <= eps && minDiff[2] == minDiffExt)
                return Vector3f(0.0f, 0.0f, -1.0f);
        } else {
            if (maxDiff[0] <= eps && maxDiff[0] == maxDiffExt)
                return Vector3f(1.0f, 0.0f, 0.0f);
            if (maxDiff[1] <= eps && maxDiff[1] == maxDiffExt)
                return Vector3f(0.0f, 1.0f, 0.0f);
            if (maxDiff[2] <= eps && maxDiff[2] == maxDiffExt)
                return Vector3f(0.0f, 0.0f, 1.0f);
        }


        std::ostringstream oss;
        oss << "Couldnt find normal for [" << std::endl
            << "  position  = " << string::indent(position) << std::endl
            << "  aabb.min = " << string::indent(aabb.min) << std::endl
            << "  aabb.max = " << string::indent(aabb.max) << std::endl
            << "  minDiff = " << string::indent(minDiff) << std::endl
            << "  maxDiff = " << string::indent(maxDiff) << std::endl
            << "]";
        Log(LogLevel::Info, oss.str().c_str());
        return Vector3f();
    }

    Vector3f reflect(Vector3f wi, Vector3f n)  const
    { 
        return wi - 2.0f * enoki::dot(wi, n) * n;
    }

    Vector3f refract(Vector3f wi, Vector3f n, Float n1, Float n2) const {
        Float eta     = n1 / n2;
        Float N_dot_I = max(-1.0, min(1.0, enoki::dot(n, wi)));
        Float k       = 1.0f - eta * eta * (1.0f - N_dot_I * N_dot_I);
        if (k < 0.0f)
            return Vector3f(0.0f);
        else
            return normalize(eta * wi - (eta * N_dot_I + enoki::sqrt(k)) * n);
    }

    bool handleNonLinearInteraction(const Scene *scene, Sampler* sampler, NonLinearInteraction &nli, SurfaceInteraction3f &si, MediumInteraction3f &mi, Ray3f &ray, Spectrum &throughput, UInt32 channel, Mask active) const override {
        // check intersection
        if (!nli.is_valid)
            return false;
        si = scene->ray_intersect(ray, active);
        if (si.t < nli.t )
            return false;

        if (!si.is_valid())
            Log(Warn, "nli 1: si became invalid ");

        //throughput *= nli.eta; // this works and should be used for correct result, but doesnt work with VRL lightcut sometimes...

        // Move ray to nli.p
        ray.o = nli.p;
        ray.mint = 0.0f;
        ray.d = nli.wo;
        ray.update();

        // update si with bent ray.
        si = scene->ray_intersect(ray, active);

        if (!si.is_valid())
        {
            Log(Warn, "nli 2: si became invalid. ray.o in bbox: %i ", bbox.contains(ray.o));

        }
        // Update mi
        mi.sh_frame                 = Frame3f(ray.d);
        mi.wi                       = -ray.d;
        auto [aabb_its, mint, maxt] = intersect_aabb(ray);
        aabb_its &= (enoki::isfinite(mint) || enoki::isfinite(maxt));
        active &= aabb_its;
        mint = max(ray.mint, mint);
        maxt = min(ray.maxt, maxt);

        auto combined_extinction = get_combined_extinction(mi, active);
        Float m                  = combined_extinction[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m, eq(channel, 1u)) = combined_extinction[1];
            masked(m, eq(channel, 2u)) = combined_extinction[2];
        } else {
            ENOKI_MARK_USED(channel);
        }

        Mask valid_mi = active && (mi.t <= maxt);
        mi.t -= nli.t;
        mi.p                                         = ray(mi.t);
        std::tie(mi.sigma_s, mi.sigma_n, mi.sigma_t) = get_scattering_coefficients(mi, valid_mi);
        mi.combined_extinction                       = combined_extinction;
        return true;
    }

    NonLinearInteraction sampleNonLinearInteraction(const Ray3f &ray, UInt32 channel, Mask active) const override { 
        NonLinearInteraction nli;
        nli.is_valid = false;

        Float rayNorm = norm(ray.d);
        if (rayNorm == 0.0f) {
            Log(LogLevel::Error, "ray with no direction");
            return nli;
        }
        
        // 1. Find AABB that ray.o resides in.
        auto[validNode, nodeIdx] = getNode(ray(ray.mint));
        if (!validNode)
        {
            nli.is_valid = false;
            return nli;
        }

        NLNode &node = grid[nodeIdx];
        nli.is_valid |= validNode;
        active &= validNode;

        nli.n1 = node.ior;
        nli.wi = ray.d;

        // 2. Intersect that AABB with the ray, find maxt intersection point.
        auto [aabb_its, mint, maxt] = node.aabb.ray_intersect(ray);
        aabb_its &= (enoki::isfinite(mint) || enoki::isfinite(maxt));
        active &= aabb_its;
        masked(mint, !active) = 0.f;
        masked(maxt, !active) = math::Infinity<Float>;
        mint = max(ray.mint, mint);
        maxt = min(ray.maxt, maxt);
        
        if (!active || maxt < math::RayEpsilon<Float>)
        {
            nli.is_valid = false;
            return nli;
        }

        nli.t = maxt + math::RayEpsilon<Float>;
        nli.p = ray(nli.t);

        // 3. Calculate n
        nli.n      = getNormal(ray(maxt), node.aabb);
        Float nDot = dot(nli.n, ray.d);
        if (nDot > 0)
        {
            if (abs(nli.n[0]) == 1.0f)
                nli.n[0] *= -1.0f;
            if (abs(nli.n[1]) == 1.0f)
                nli.n[1] *= -1.0f;
            if (abs(nli.n[2]) == 1.0f)
                nli.n[2] *= -1.0f;
        }

        // 4. Find neighbouring node that was hit, fill in n2.
        //Point3f neighbourOrigin = nli.p - nli.n * math::RayEpsilon<Float>;
        auto [validNeighbour, neighbourIdx] = getNeighbour(nodeIdx, nli.n);

        if (!validNeighbour) {
            nli.is_valid = false;
            return nli;
        }
        NLNode &neighbour = grid[neighbourIdx];

        nli.n2 = neighbour.ior;

        // 5. Refract ray using nli info
        nli.wo = refract(nli.wi, nli.n, nli.n1, nli.n2);
        if (norm(nli.wo) == 0.0f) {
            nli.eta = 1.0f;
            nli.t   = maxt;
            nli.p   = ray(nli.t);
            nli.wo = reflect(nli.wi, nli.n);
        } else {
            Float cos_theta_i = Frame3f::cos_theta(-nli.wi);
            auto outside_mask = cos_theta_i >= 0.f;
            Float eta         = nli.n1 / nli.n2;
            Float rcp_eta     = rcp(eta);
            nli.eta           = select(outside_mask, eta, rcp_eta);
        }

        // Update mi since info should be gathered from a different point
        if (norm(nli.wo) == 0.0f || (norm(nli.n) == 0.0f) || nodeIdx == neighbourIdx) {
            nli.wo = ray.d;

            std::ostringstream oss;
            oss << "Found node for origin[" << std::endl
                << "  ray.o  = " << string::indent(ray.o) << std::endl
                << "  ray.d  = " << string::indent(ray.d) << std::endl
                << "  mint  = " << string::indent(mint) << std::endl
                << "  maxt  = " << string::indent(maxt) << std::endl
                << "  nodeIdx  = " << string::indent(nodeIdx) << std::endl
                << "  nbIdx  = " << string::indent(neighbourIdx) << std::endl
                << "  nli.p  = " << string::indent(nli.p) << std::endl
                << "  nli.wo  = " << string::indent(nli.wo) << std::endl
                << "  nli.n  = " << string::indent(nli.n) << std::endl
                << "  node.aabb = " << string::indent(node.aabb) << std::endl
                << "  neigh.aabb = " << string::indent(neighbour.aabb) << std::endl
                << "  n1  = " << string::indent(nli.n1) << std::endl
                << "  n2 = " << string::indent(nli.n2) << std::endl
                << "]";
            Log(Info, oss.str().c_str());
            Log(Warn, "neighbour and node ar ethe same! or normal is 0");
        }

        return nli; 
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
            << "  cellSize = " << string::indent(cellSize) << std::endl
            << "  resolution = " << string::indent(resolution) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    ref<Volume> m_sigmat, m_albedo;
    NLNode *grid;
    int arraySize;
    int iorMethod;
    Vector3f resolution;
    Float topIoR, bottomIoR;
    Vector3f bottomTemperature;
    Vector3f topTemperature;
    Vector3f bottomPressure;
    Vector3f topPressure;
    Point3f cellSize;
    bool m_from_bottom;
};

MTS_IMPLEMENT_CLASS_VARIANT(NonLinearMedia, Medium)
MTS_EXPORT_PLUGIN(NonLinearMedia, "Nonlinear Homogeneous Medium")
NAMESPACE_END(mitsuba)
