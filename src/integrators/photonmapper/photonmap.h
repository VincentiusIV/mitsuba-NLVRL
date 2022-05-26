#pragma once
#include <enoki/stl.h>
#include <enoki/fwd.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <random>
#include "kdtree.h"
#include "../vrl/vrl_struct.h"


NAMESPACE_BEGIN(mitsuba)

#define INV_PI 0.31830988618379067154
#define UNIT_SPHERE_VOLUME 4.18879020478639

template <typename Float, typename Spectrum> struct PhotonData {
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    Spectrum power;
    Normal3f normal;
    Vector3f direction;
    int depth;

    inline PhotonData() : power(0.0f), normal(0.0f), direction(0.0f), depth(0) {}

    PhotonData(const Normal3f &normal,
           const Vector3f &direction, const Spectrum &spectrum,
           const int &depth) {
        this->power  = spectrum;
        this->normal    = normal;
        this->direction = direction;
        this->depth     = depth;
    }

};

template <typename Float, typename Spectrum, typename Point>
struct Photon : SimpleKDNode<Point, PhotonData<Float, Spectrum>> {
    typedef PhotonData<Float, Spectrum> PhotonData;

    inline Photon() {}

    inline Photon(const Point &position, const PhotonData &photon)
        : SimpleKDNode(photon) {
        this->setPosition(position);
    }
};

template <typename Float, typename Spectrum>
class PhotonMap {
public:
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef PhotonData<Float, Spectrum> PhotonData;
    typedef Photon<Float, Spectrum, Point3f> Photon;
    typedef PointKDTree<Float, Spectrum, Photon> PhotonTree;
    typedef typename PhotonTree::IndexType IndexType;
    typedef typename PhotonTree::SearchResult SearchResult;

    PhotonMap(size_t photonCount) {
        Log(LogLevel::Info, "Constructing PhotonMap...");
        m_scale = 1.0f;
        m_kdtree.reserve(photonCount);
    }

    
    inline void clear() { m_kdtree.clear(); }
    /// Resize the kd-tree array
    inline void resize(size_t size) { m_kdtree.resize(size); }
    /// Reserve a certain amount of memory for the kd-tree array
    inline void reserve(size_t size) { m_kdtree.reserve(size); }
    /// Return the size of the kd-tree
    inline size_t size() const { return m_kdtree.size(); }
    /// Return the capacity of the kd-tree
    inline size_t capacity() const { return m_kdtree.capacity(); }
    /// Append a kd-tree photonData to the photonData array
    inline void push_back(const Photon &photon) { m_kdtree.push_back(photon); }
    /// Return one of the photons by index
    inline Photon &operator[](size_t idx) { return m_kdtree[idx]; }
    /// Return one of the photons by index (const version)
    inline const Photon &operator[](size_t idx) const { return m_kdtree[idx];
    }

    inline void setScaleFactor(Float value) { m_scale = value; }

    inline float getScaleFactor() const { return m_scale; }

    inline size_t getDepth() const { return m_kdtree.getDepth(); }

    inline void build(bool recomputeAABB = false) {
        m_kdtree.build(recomputeAABB);
    }

    inline size_t nnSearch(const Point3f &p, Float &sqrSearchRadius, size_t k,
                           SearchResult *results) const {
        return m_kdtree.nnSearch(p, sqrSearchRadius, k, results);
    }

    inline size_t nnSearch(const Point3f &p, size_t k, SearchResult *) const {
        return m_kdtree.nnSearch(p, k, results);
    }

    inline void insert(const Point3f &position, const PhotonData &photon) {
        Photon newNode(position, photon);
        push_back(newNode);
        /*std::ostringstream stream;
        stream << "Inserting photon, power = " << photon.power << ", depth:" << photon.depth;
        std::string str = stream.str();
        Log(LogLevel::Info, str.c_str());*/
    }

    Spectrum estimateRadiance(const SurfaceInteraction3f &si,
                              float searchRadius, size_t maxPhotons, bool smooth = false) const {
        SearchResult *results  = new SearchResult[maxPhotons]; // this is really expensive, consider a buffer per thread
        float squaredRadius = searchRadius * searchRadius;
        size_t resultCount  = nnSearch(si.p, squaredRadius, maxPhotons, results);
        float invSquaredRadius = 1.0f / squaredRadius;
        Spectrum result(0.0f);
        const BSDF *bsdf = si.bsdf();
        Spectrum bsdfVal;
        for (size_t i = 0; i < resultCount; i++) {
            const SearchResult &searchResult = results[i];
            const Photon &photon         = m_kdtree[searchResult.index];
            const PhotonData &photonData             = photon.getData();

            Vector3f wo = si.wi;
            SurfaceInteraction3f mutesi(si);
            mutesi.wi = si.to_local(-photonData.direction);

            BSDFContext bRec(TransportMode::Importance);
            bsdfVal = bsdf->eval(bRec, si, mutesi.wi);
            bsdfVal = si.to_world_mueller(bsdfVal, -wo, si.wi);

            if (smooth) {
                Float sqrTerm = 1.0f - searchResult.distSquared * invSquaredRadius;
                bsdfVal *= (sqrTerm * sqrTerm);
            }

            result += photonData.power * bsdfVal;
        }

        delete[] results;
        result *= m_scale * INV_PI * invSquaredRadius;
        if (smooth)
            result *= 3;

        return result;
    }

    Spectrum estimateCausticRadiance(const SurfaceInteraction3f &si, float searchRadius, size_t maxPhotons) const {
        SearchResult *results  = new SearchResult[maxPhotons]; // this is really expensive, consider a buffer per thread
        float squaredRadius    = searchRadius * searchRadius;
        size_t resultCount     = nnSearch(si.p, squaredRadius, maxPhotons, results);
        float invSquaredRadius = 1.0f / squaredRadius;
        Spectrum result(0.0f);
        const BSDF *bsdf = si.bsdf();
        Spectrum bsdfVal;
        for (size_t i = 0; i < resultCount; i++) {
            const SearchResult &searchResult = results[i];
            const Photon &photon             = m_kdtree[searchResult.index];
            const PhotonData &photonData     = photon.getData();

            Vector3f wo = si.wi;
            SurfaceInteraction3f mutesi(si);
            mutesi.wi = si.to_local(-photonData.direction);

            BSDFContext bRec(TransportMode::Importance);
            bsdfVal = bsdf->eval(bRec, si, mutesi.wi);
            bsdfVal = si.to_world_mueller(bsdfVal, -wo, si.wi);

            double wp = max(0.0, 1.0f - sqrt(searchResult.distSquared * invSquaredRadius));

            result += photonData.power * bsdfVal * wp;
        }

        delete[] results;
        result *= 3.0 * m_scale * INV_PI * invSquaredRadius;

        return result;
    }

    float K2(float x) const { 
        return 3 * INV_PI * (1 - x * x) * (1 - x * x);
    }

    Spectrum estimateRadianceVolume(Point3f gatherPoint, Vector3f wo, const Medium *medium, Sampler *sampler, float searchRadius, size_t maxPhotons) const {
        SearchResult *results  = new SearchResult[maxPhotons];
        float squaredRadius    = searchRadius * searchRadius;
        size_t resultCount     = nnSearch(gatherPoint, squaredRadius, maxPhotons, results);
        float invSquaredRadius = 1.0f / squaredRadius;

        Spectrum result(0.0f);

        MediumInteraction3f mi1;
        PhaseFunctionContext phase_ctx1(sampler);
        const PhaseFunction *pf = medium->phase_function();

        for (size_t i = 0; i < resultCount; i++) {
            const SearchResult &searchResult = results[i];
            const Photon &photon             = m_kdtree[searchResult.index];
            const PhotonData &photonData     = photon.getData();

            mi1.wi = -photonData.direction;
            Float photonPF = pf->eval(phase_ctx1, mi1, wo);

            result += photonData.power * photonPF;
        }
        result *= m_scale;
        result /= (UNIT_SPHERE_VOLUME * searchRadius * searchRadius * searchRadius);

        delete[] results;
        return result;
    }

protected:
    PhotonTree m_kdtree;
    float m_scale;
};

NAMESPACE_END(mitsuba)