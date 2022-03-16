#pragma once

#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/core/timer.h>

#include "photonmap.h"

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Implements the beam radiance estimate described in
 * "The Beam Radiance Estimator for Volumetric Photon Mapping"
 * by Wojciech Jarosz, Matthias Zwicker, and Henrik Wann Jensen.
 */

template <typename Float, typename Spectrum>
class BeamRadianceEstimator {
public:
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    typedef typename PhotonMap::IndexType IndexType;

    /**
     * \brief Create a BRE acceleration data structure from
     * an existing volumetric photon map
     */
    BeamRadianceEstimator(const PhotonMap* pmap, size_t lookupSize) {
        /* Use an optimization proposed by Jarosz et al, which accelerates
       the radius computation by extrapolating radius information obtained
       from a kd-tree lookup of a smaller size */
        size_t reducedLookupSize = (size_t) std::sqrt((Float) lookupSize);
        Float sizeFactor = (Float) lookupSize / (Float) reducedLookupSize;

        m_photonCount = pmap->size();
        m_scaleFactor = pmap->getScaleFactor();
        m_depth       = pmap->getDepth();

        Log(LogLevel::Info, "Allocating memory for the BRE acceleration data structure");
        m_nodes = new BRENode[m_photonCount];

        Log(LogLevel::Info, "Computing photon radii ..");
        int tcount = 1;
        PhotonMap::SearchResult **resultsPerThread =
            new PhotonMap::SearchResult *[tcount];
        for (int i = 0; i < tcount; ++i)
            resultsPerThread[i] = new PhotonMap::SearchResult[reducedLookupSize + 1];

            for (int i = 0; i < (int) m_photonCount; ++i) {
                int tid = 0;

            PhotonMap::SearchResult *results = resultsPerThread[tid];
            const Photon &photon             = pmap->operator[](i);
            BRENode &node                    = m_nodes[i];
            node.photon                      = photon;

            Float searchRadiusSqr = std::numeric_limits<Float>::infinity();
            pmap->nnSearch(photon.p, searchRadiusSqr, reducedLookupSize, results);

            /* Compute photon radius based on a locally uniform density
             * assumption */
            node.radius = std::sqrt(searchRadiusSqr * sizeFactor);
        }
        Log(LogLevel::Info, "Done ");

        Log(LogLevel::Info, "Generating a hierarchy for the beam radiance estimate");

        buildHierarchy(0);
        Log(LogLevel::Info, "Done ");

        for (int i = 0; i < tcount; ++i)
            delete[] resultsPerThread[i];
        delete[] resultsPerThread;
    }

    /// Compute the beam radiance estimate for the given ray segment and medium
    Spectrum query(const Ray& ray, const Medium* medium) const {
        const Ray ray(r(r.mint), r.d, 0, r.maxt - r.mint, r.time);
        IndexType *stack =
            (IndexType *) alloca((m_depth + 1) * sizeof(IndexType));
        IndexType index = 0, stackPos = 1;
        Spectrum result(0.0f);

        const Spectrum &sigmaT     = medium->getSigmaT();
        const PhaseFunction *phase = medium->getPhaseFunction();
        MediumSamplingRecord mRec;

        while (stackPos > 0) {
            const BRENode &node  = m_nodes[index];
            const Photon &photon = node.photon;

            /* Test against the node's bounding box */
            Float mint, maxt;
            if (!node.aabb.rayIntersect(ray, mint, maxt) || maxt < ray.mint ||
                mint > ray.maxt) {
                index = stack[--stackPos];
                continue;
            }

            /* Recurse on inner photons */
            if (!photon.isLeaf()) {
                if (hasRightChild(index))
                    stack[stackPos++] = photon.getRightIndex(index);
                index = photon.getLeftIndex(index);
            } else {
                index = stack[--stackPos];
            }

            Vector originToCenter = node.photon.getPosition() - ray.o;
            Float diskDistance    = dot(originToCenter, ray.d),
                  radSqr          = node.radius * node.radius;
            Float distSqr =
                (ray(diskDistance) - node.photon.getPosition()).lengthSquared();

            if (diskDistance > 0 && distSqr < radSqr) {
                Float weight = K2(distSqr / radSqr) / radSqr;

                Vector wi = -node.photon.getDirection();

                Spectrum transmittance = Spectrum(-sigmaT * diskDistance).exp();
                result +=
                    transmittance * node.photon.getPower() *
                    phase->eval(PhaseFunctionSamplingRecord(mRec, wi, -ray.d)) *
                    (weight * m_scaleFactor);
            }
        }

        return result;
    }

protected:
    /// Release all memory
    virtual ~BeamRadianceEstimator() { delete[] m_nodes; }

    /// Fit a hierarchy of bounding boxes to the stored photons
    AABB buildHierarchy(IndexType index) {
        BRENode &node = m_nodes[index];

        Point center = node.photon.getPosition();
        Float radius = node.radius;
        node.aabb    = AABB(center - Vector(radius, radius, radius),
                         center + Vector(radius, radius, radius));

        if (!node.photon.isLeaf()) {
            IndexType left  = node.photon.getLeftIndex(index);
            IndexType right = node.photon.getRightIndex(index);
            if (left)
                node.aabb.expandBy(buildHierarchy(left));
            if (right)
                node.aabb.expandBy(buildHierarchy(right));
        }

        return node.aabb;
    }

    /// Blurring kernel used by the BRE
    inline Float K2(Float sqrParam) const {
        Float tmp = 1-sqrParam;
        return (3/M_PI) * tmp * tmp;
    }

    /**
     * \brief Return whether or not the inner node of the
     * specified index has a right child node.
     *
     * This function is available for convenience and abstracts away some
     * details about the underlying node representation.
     */
    inline bool hasRightChild(IndexType index) const {
        if (Photon::leftBalancedLayout) {
            return 2*index+2 < m_photonCount;
        } else {
            return m_nodes[index].photon.getRightIndex(index) != 0;
        }
    }
protected:
    struct BRENode {
        AABB aabb;
        Photon photon;
        Float radius;
    };

    BRENode *m_nodes;
    Float m_scaleFactor;
    size_t m_photonCount;
    size_t m_depth;
};

NAMESPACE_END(mitsuba)
