#pragma once

#include <mitsuba/core/bbox.h>
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
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    typedef PhotonMap<Float, Spectrum> PhotonMap;
    typedef typename PhotonMap::IndexType IndexType;
    typedef typename PhotonMap::Photon Photon;
    typedef typename Photon::PointType PointType;
    typedef BoundingBox<PointType> AABB;

    /**
     * \brief Create a BRE acceleration data structure from
     * an existing volumetric photon map
     */
    inline BeamRadianceEstimator() { }

    void build(const PhotonMap* pmap, size_t lookupSize, Float scaleVol = 1.f, bool cameraHeuristic = false) {
        /* Use an optimization proposed by Jarosz et al, which accelerates
       the radius computation by extrapolating radius information obtained
       from a kd-tree lookup of a smaller size */
        size_t reducedLookupSize = (size_t) std::sqrt((Float) lookupSize);
        Float sizeFactor = (Float) lookupSize / (Float) reducedLookupSize;

        m_photonCount = pmap->size();
        m_scaleFactor = pmap->getScaleFactor();
        m_depth       = pmap->getDepth();

        Log(LogLevel::Info,
            "Allocating memory for the BRE acceleration data structure");
        m_nodes = new BRENode[m_photonCount];

        Log(LogLevel::Info, "Computing photon radii ..");
        int tcount = 1;
        PhotonMap::SearchResult **resultsPerThread =
            new PhotonMap::SearchResult *[tcount];
        for (int i = 0; i < tcount; ++i)
            resultsPerThread[i] =
                new PhotonMap::SearchResult[reducedLookupSize + 1];

        for (int i = 0; i < (int) m_photonCount; ++i) {
            int tid = 0;

            PhotonMap::SearchResult *results = resultsPerThread[tid];
            const Photon &photon             = pmap->operator[](i);
            BRENode &node                    = m_nodes[i];
            node.photon                      = photon;

            Float searchRadiusSqr = std::numeric_limits<Float>::infinity();
            pmap->nnSearch(photon.position, searchRadiusSqr, reducedLookupSize,
                           results);

            /* Compute photon radius based on a locally uniform density
             * assumption */
            node.radius = std::sqrt(searchRadiusSqr * sizeFactor);
        }
        Log(LogLevel::Info, "Done ");

        Log(LogLevel::Info,
            "Generating a hierarchy for the beam radiance estimate");

        buildHierarchy(0);
        Log(LogLevel::Info, "Done ");

        for (int i = 0; i < tcount; ++i)
            delete[] resultsPerThread[i];
        delete[] resultsPerThread;
    }

    /// Compute the beam radiance estimate for the given ray segment and medium
    Spectrum query(const Ray3f &r, const Medium *medium, const SurfaceInteraction3f &si, Sampler *sampler,
                   UInt32 channel, Mask active, int maxDepth,
                   bool use3Dkernel) const {
        const Ray3f ray(r, 0, r.maxt - r.mint);
        IndexType *stack = new IndexType[m_depth + 1];
        IndexType index = 0, stackPos = 1;
        Spectrum result(0.0f);

        MediumInteraction3f mi = medium->sample_interaction(ray, sampler->next_1d(), channel, active);
        auto [sigma_s, sigma_n, sigmaT] = medium->get_scattering_coefficients(mi);
        const PhaseFunction *phase = medium->phase_function();

        while (stackPos > 0) {
            const BRENode &node  = m_nodes[index];
            const Photon &photon = node.photon;

            /* Test against the node's bounding box */
            auto[active, mint, maxt] = node.aabb.ray_intersect(ray);
            if (!any_or<true>(active) || maxt < ray.mint || mint > ray.maxt) {
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

            if (maxDepth != -1 && node.photon.getData().depth > maxDepth) {
                continue;
            }

            Vector3f originToCenter = node.photon.getPosition() - ray.o;
            Float diskDistance    = dot(originToCenter, ray.d),
                  radSqr          = node.radius * node.radius;
            Vector3f between = (ray(diskDistance) - node.photon.getPosition());
            Float distSqr = between[0] * between[0] + between[1] * between[1] +
                            between[2] * between[2];

            if (diskDistance > 0 && distSqr < radSqr) {
                if (use3Dkernel) {
                    if (diskDistance - (node.radius * 2) > ray.maxt) {
                        // Try to stop the gathering early
                        continue;
                    }

                    Float weight = 1 / ((4.0 / 3.0) * M_PI * std::pow(node.radius, 3));
                    // Basically splat the photon enery to the camera segment
                    // that intersects the 3D kernel

                    // Determine delta T
                    Float deltaT     = std::sqrt(radSqr - distSqr);
                    Float tminKernel = diskDistance - deltaT;
                    Float diskDistanceRand =
                        tminKernel +
                        2 * deltaT *
                            sampler->next_1d(); // the segment tc- and tc+

                    if (diskDistanceRand < 0 || diskDistanceRand > ray.maxt) {
                        // No contribution, go to the next
                        continue;
                    }

                    Float invPdfSampling = std::max(2.0f * deltaT, (Float) 0.0001f);

                    Vector3f wi = -node.photon.getData().direction;

                    Ray3f baseRay(ray);
                    baseRay.maxt = diskDistanceRand;
                    MediumInteraction3f mRecBase = medium->sample_interaction(
                        baseRay, sampler->next_1d(), channel, active);
                    std::pair<UnpolarizedSpectrum, UnpolarizedSpectrum> tr_and_pdf =
                            medium->eval_tr_and_pdf(mRecBase, si, active);

                    PhaseFunctionContext pctx(sampler);
                    result += tr_and_pdf.first * node.photon.getData().power 
                        * phase->eval(pctx, mi, -ray.d) 
                        * (weight * m_scaleFactor) * invPdfSampling;

                } else {
                    Float weight = 1 / (M_PI * std::pow(node.radius, 2));

                    if (diskDistance > ray.maxt) {
                        continue; // Cannot gather it
                    }

                    Vector3f wi = -node.photon.getData().direction;

                    Ray3f baseRay(ray);
                    baseRay.maxt = diskDistance;
                    MediumInteraction3f mRecBase = medium->sample_interaction(
                        baseRay, sampler->next_1d(), channel, active);
                    std::pair<UnpolarizedSpectrum, UnpolarizedSpectrum> tr_and_pdf = medium->eval_tr_and_pdf(mRecBase, si, active);

                    PhaseFunctionContext pctx(sampler);
                    result += tr_and_pdf.first * node.photon.getData().power 
                        * phase->eval(pctx, mi, -ray. d) 
                        * (weight * m_scaleFactor);
                }
            }
        }
        delete[] stack;
        return result;
    }

protected:
    /// Release all memory
    virtual ~BeamRadianceEstimator() { delete[] m_nodes; }

    /// Fit a hierarchy of bounding boxes to the stored photons
    AABB buildHierarchy(IndexType index) {
        BRENode &node = m_nodes[index];

        Point3f center = node.photon.getPosition();
        Float radius = node.radius;
        node.aabb      = AABB(center - Vector3f(radius, radius, radius),
                         center + Vector3f(radius, radius, radius));

        if (!node.photon.isLeaf()) {
            IndexType left  = node.photon.getLeftIndex(index);
            IndexType right = node.photon.getRightIndex(index);
            if (left)
                node.aabb.expand(buildHierarchy(left));
            if (right)
                node.aabb.expand(buildHierarchy(right));
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
