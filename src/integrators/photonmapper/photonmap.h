#pragma once
#include <enoki/stl.h>
#include <enoki/fwd.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <random>
#include "kdtree.h"

NAMESPACE_BEGIN(mitsuba)

#define INV_PI 0.31830988618379067154

template <typename Float, typename Spectrum> struct Photon {
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    Spectrum spectrum;
    Normal3f normal;
    Vector3f direction;
    int depth;

    inline Photon() : spectrum(0.0f), normal(0.0f), direction(0.0f), depth(0) {}

    Photon(const Normal3f &normal,
           const Vector3f &direction, const Spectrum &spectrum,
           const int &depth) {
        this->spectrum  = spectrum;
        this->normal    = normal;
        this->direction = direction;
        this->depth     = depth;
    }
};

template <typename Float, typename Spectrum, typename Point>
struct PhotonNode : SimpleKDNode<Point, Photon<Float, Spectrum>> {
    typedef Photon<Float, Spectrum> Photon;

    inline PhotonNode() : SimpleKDNode() {}

    inline PhotonNode(const Point &position, const Photon &photon)
        : SimpleKDNode(photon) {
        this->setPosition(position);
    }
};

template <typename Float, typename Spectrum>
class PhotonMap {
public:
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    typedef Photon<Float, Spectrum> Photon;
    typedef PhotonNode<Float, Spectrum, Point3f> PhotonNode;
    typedef PointKDTree<Float, Spectrum, PhotonNode> PhotonTree;
    typedef typename PhotonTree::IndexType IndexType;
    typedef typename PhotonTree::SearchResult SearchResult;

    PhotonMap(size_t photonCount) {
        Log(LogLevel::Info, "Constructing PhotonMap...");
        m_scale = 0.001f;
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
    /// Append a kd-tree photon to the photon array
    inline void push_back(const PhotonNode &photon) { m_kdtree.push_back(photon); }
    /// Return one of the photons by index
    inline PhotonNode &operator[](size_t idx) { return m_kdtree[idx]; }
    /// Return one of the photons by index (const version)
    inline const PhotonNode &operator[](size_t idx) const { return m_kdtree[idx];
    }

    /// Perform a nearest-neighbor query, see \ref PointKDTree for details
    inline size_t nnSearch(const Point3f &p, Float &sqrSearchRadius, size_t k,
                           std::vector<SearchResult> &results) const {
        return m_kdtree.nnSearch(p, sqrSearchRadius, k, results);
    }

    /// Perform a nearest-neighbor query, see \ref PointKDTree for details
    inline size_t nnSearch(const Point3f &p, size_t k,
                           std::vector<SearchResult> &results) const {
        return m_kdtree.nnSearch(p, k, results);
    }

    inline void insert(const Point3f &position, const Photon &photon) {
        PhotonNode newNode(position, photon);
        push_back(newNode);
        //push_back(newNode);
        std::string numPhotonsStr = "Inserted new photon, photon Count: " +
                                    std::to_string(m_kdtree.size());
        Log(LogLevel::Info, numPhotonsStr.c_str());   

    }

    Spectrum estimateRadiance(const SurfaceInteraction3f &si,
                              float searchRadius, size_t maxPhotons) const {
        std::vector<SearchResult> results;
        results.reserve(maxPhotons);
        float squaredRadius = searchRadius * searchRadius;
        size_t resultCount  = nnSearch(si.p, squaredRadius, maxPhotons, results);
        float invSquaredRadius = 1.0f / squaredRadius;

        std::string numPhotonsStr =
            "- nn Count: " + std::to_string(resultCount);
        //Log(LogLevel::Info, numPhotonsStr.c_str());    

        Spectrum result(0.0f);
        const BSDF *bsdf = si.bsdf();
        for (size_t i = 0; i < resultCount; i++) {
            const SearchResult &searchResult = results[i];
            const PhotonNode &photonNode     = m_kdtree[searchResult.index];
            const Photon &photon             = photonNode.getData();
            Vector3f between      = photonNode.getPosition() - si.p;
            float sqrTerm = 1.0f - searchResult.distSquared * invSquaredRadius;
            if (searchResult.distSquared > squaredRadius)
                continue;
            Vector3f wi = si.to_local(-photon.direction);
            BSDFContext bRec(TransportMode::Importance);
            result += photon.spectrum * bsdf->eval(bRec, si, si.wi) * (sqrTerm * sqrTerm);
        }

        return result * (m_scale * 3.0 * INV_PI * invSquaredRadius);
    }

    Spectrum estimateIrradiance(const Point3f &p, const Normal3f &n,
                                float searchRadius, int maxDepth,
                                size_t maxPhotons) const {
        std::vector<SearchResult> results;
        results.reserve(maxPhotons);
        float squaredRadius = searchRadius * searchRadius;
        size_t resultCount = nnSearch(p, squaredRadius, maxPhotons, results);
        float invSquaredRadius = 1.0f / squaredRadius;
        Spectrum result(0.0f);
        for (size_t i = 0; i < resultCount; i++) {
            const SearchResult &searchResult = results[i];
            const PhotonNode &photonNode     = m_kdtree[searchResult.index];
            const Photon &photon              = photonNode.getData();
            if (photon.depth > maxDepth)
                continue;

            Vector3f between  = photonNode.position - p;
            float distSquared = between[0] * between[0] +
                                between[1] * between[1] +
                                between[2] * between[2];
            float sqrTerm = 1.0f - distSquared * invSquaredRadius;

            if (distSquared > squaredRadius)
                continue;

            Vector3f wi     = -photon.direction;
            float wiDotGeoN = dot(photon.normal, wi);
            float wiDotShN  = dot(n, wi);

            if (dot(wi, n) > 0 && dot(photon.normal, n) > 1e-1f &&
                wiDotGeoN > 1e-2f) {
                Spectrum power =
                    photon.spectrum * std::abs(wiDotShN / wiDotGeoN);

                float sqrTerm = 1.0f - distSquared * invSquaredRadius;

                result += power * (sqrTerm * sqrTerm);
            }
        }
        return result * (m_scale * 3.0 * INV_PI * invSquaredRadius);
    }

protected:
    PhotonTree m_kdtree;
    float m_scale;
};

NAMESPACE_END(mitsuba)