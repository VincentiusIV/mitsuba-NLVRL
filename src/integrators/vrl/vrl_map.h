#pragma once

#include <fstream>

#include "vrl_lightcut.h"
#include "vrl_struct.h"
#include <enoki/fwd.h>
#include <enoki/stl.h>

NAMESPACE_BEGIN(mitsuba)

enum EVRLAcceleration { ENoVRLAcceleration, ELightCutAcceleration };

enum EVRLRussianRoulette {
    ENoRussianRoulette,
    EDistanceRoulette,
    ETransmittanceRoulette,
};

template <typename Float, typename Spectrum> class VRLMap {
public:
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    typedef VRL<Float, Spectrum> VRL;
    typedef VRLLightCut<Float, Spectrum> VRLLightCut;

    VRLMap(size_t nbVRL) {
        m_map.reserve(nbVRL);
        m_maxSize = nbVRL;
        m_accel = ENoVRLAcceleration;

        // MemoryPool memPool;
        // m_map.clear();
        // while (m_map.size() < nbVRL) {
        //    Path p;
        //    p.initialize(scene, 0.f, EImportance, memPool);
        //    p.randomWalk(scene, sensorSampler, maxDepth, rrDepth, EImportance, memPool);
        //    // Transform path into VPL
        //    // Note that if the rendering type is only medium to medium
        //    // Only relevant VPL are generated
        //    Spectrum weight(1.f);
        //    for (int idVertex = 0; idVertex < p.vertexCount() - 1; idVertex++) {
        //        // Update the weight this the medium (potentially)
        //        const PathVertex *vertex = p.vertex(idVertex);
        //        const PathEdge *edge     = p.edge(idVertex);
        //        const Medium *medium     = edge->medium;
        //        weight *= vertex->rrWeight * vertex->weight[EImportance];
        //        if (idVertex != 0 && medium != nullptr && edge->length != 0 && (!weight.isZero())) {
        //            VRL vrl(vertex->getPosition(), medium, weight, idVertex);
        //            Point3f next_point = vertex->getPosition() + edge->d * edge->length;
        //            vrl.setEndPoint(next_point);
        //            m_map.emplace_back(std::move(vrl));
        //        }
        //        weight *= p.edge(idVertex)->weight[EImportance];
        //    } // Add vpl contribution
        //    p.release(memPool);
        //    m_scale += 1;
        //}
    }

    bool can_add() {
        return (size() < m_maxSize);
    }

    bool push_back(VRL &vrl, bool log) {
        if (!can_add())
            return false;
        if (vrl.getMedium() == nullptr)
            return false;
        if (vrl.flux == Spectrum(0.0))
            return false;
        /*std::ostringstream stream;
        if (log)
            stream << "Insert VRL:" << vrl;                
        std::string str = stream.str();
        Log(LogLevel::Info, str.c_str());*/
        m_map.emplace_back(vrl);
        return true;
    }

    virtual ~VRLMap() {}

    void toLong(const Scene *scene) {
        // Normally we generate short beams, but by extend the beam (to the surface intersection)
        // We can change short beams to long one
        for (size_t i = 0; i < m_map.size(); i++) {
            m_map[i].convertToLong(scene);
        }
    }

    void dumpVRL(const std::string &filename) {
        std::ofstream file("lc.obj");
        file << "o VRL\n";
        for (auto &vrl : m_map) {
            file << "v " << vrl.origin.x << " " << vrl.origin.y << " " << vrl.origin.z << "\n";
            auto end = vrl.getEndPoint();
            file << "v " << end.x << " " << end.y << " " << end.z << "\n";
        }

        for (auto i = 0; i < m_map.size(); i++) {
            file << "f " << 1 + i * 2 << " " << 2 + i * 2 << "\n";
        }
        file.close();
    }

    void build(const Scene *scene, EVRLAcceleration accel, Sampler *sampler, int thresholdBetterDist, Float thresholdError) {
        m_accel = accel;
        std::ostringstream stream;
        stream << "Building VRL map... Scale factor = " << m_scale;
        std::string str = stream.str();
        Log(LogLevel::Info, str.c_str());
        if (m_accel == ENoVRLAcceleration) {
            Log(LogLevel::Info, "No VRL acceleration.");
            // Nothing to do
        } else if (m_accel == ELightCutAcceleration) {
            Log(LogLevel::Info, "Building VRL lightcut acceleration.");
            m_lc = new VRLLightCut(scene, m_map, sampler, thresholdBetterDist, thresholdError);
        } else {
            Log(LogLevel::Error, "build for acceleration is not implemented");
        }
    }

    void dicingVRL(const Scene *scene, Sampler *sampler, size_t avgNumberDice) {
        // Compute the average length
        auto avg_length = 0.0;
        for (auto &v : m_map) {
            avg_length += v.length;
        }
        avg_length /= m_map.size();
        Log(LogLevel::Info, "Average length: %f", avg_length);

        // Generate a new map with the diced version of VRL
        std::vector<VRL> dicedVRL;
        dicedVRL.reserve(m_map.size() * (avg_length + 1));
        // TODO: Might be not optimized
        for (auto &v : m_map) {
            auto new_vrls = v.dice(scene, sampler, avg_length / avgNumberDice);
            for (auto v2 : new_vrls) {
                dicedVRL.push_back(v2);
            }
        }

        // Replace the map with the diced VRL one
        std::string debugStr = "Number of VRLs: " + std::to_string(dicedVRL.size());
        Log(LogLevel::Info, debugStr.c_str());
        m_map = dicedVRL;
    }

    // TODO: Fix the specific parameters for RR

    // returns nb_evaluation, color, nb_BBIntersection
    std::tuple<size_t, Spectrum, size_t> query(const Ray3f &ray, const Scene *scene, Sampler *sampler, int renderScatterDepth, Float lengthOfRay, bool useUniformSampling, bool useDirectIllum, const EVRLRussianRoulette strategyRR, Float scaleRR, UInt32 samples, UInt32 channel) const {
        if (m_map.size() == 0)
            return { 0, Spectrum(0.0), 0 };

        Spectrum Li(0.0);
        size_t nb_evaluation     = 0;
        size_t nb_BBIntersection = 0;

        for (size_t i = 0; i < samples; i++) {
            if (m_accel == ENoVRLAcceleration) {
                size_t nbVRLPruned = 0;
                for (const VRL &vrl : m_map) {

                    // filter ray by scatterdepth, set to -1 (default) for no filtering
                    if (renderScatterDepth != -1) {
                        if (vrl.getScatterDepth() != renderScatterDepth) {
                            continue;
                        }
                    }
                    if (strategyRR == ENoRussianRoulette) {
                        Spectrum contrib = vrl.getContrib(scene, useUniformSampling, useDirectIllum, ray, lengthOfRay, sampler, channel) * m_scale;

                        if (std::isnan(contrib[0]) || std::isnan(contrib[1]) || std::isnan(contrib[2]) || std::isinf(contrib[0]) || std::isinf(contrib[1]) || std::isinf(contrib[2])) {
                            continue;
                        }
                        /*std::ostringstream stream;
                        stream << "Contrib of a vrl: " << vrl << " = " << contrib;
                        std::string str = stream.str();
                        Log(LogLevel::Info, str.c_str());*/
                        Li += contrib;
                        nb_evaluation += 1;
                    } else if (strategyRR == EDistanceRoulette) {
                        auto computeMinRayToRayDistance = [&]() -> Float {
                            auto res = vrl.findClosetPoint(ray);
                            return enoki::squared_norm((ray.o + ray.d * res.tCam) - (vrl.origin + vrl.direction * res.tVRL));
                        };

                        // Minimal survival rate of 5%
                        Float min_distance_sqr = sqrt(computeMinRayToRayDistance());
                        Float rrWeight         = min(1 / (min_distance_sqr * scaleRR), 1.0);
                        if (rrWeight > sampler->next_1d()) {
                            Spectrum contrib = (vrl.getContrib(scene, useUniformSampling, useDirectIllum, ray, lengthOfRay, sampler, channel) / rrWeight) * m_scale;
                            if (std::isnan(contrib[0]) || std::isnan(contrib[1]) || std::isnan(contrib[2]) || std::isinf(contrib[0]) || std::isinf(contrib[1]) || std::isinf(contrib[2])) {
                                continue;
                            }
                            Log(LogLevel::Info, "found a contrib i guess");
                            Li += contrib;
                            nb_evaluation += 1;
                        } else {
                            nbVRLPruned += 1;
                        }
                    } else {
                        Log(LogLevel::Error, "This russian roulette scheme is not implemented");
                    }
                }

                /*VRLPercentagePruned += nbVRLPruned;
                VRLPercentagePruned.incrementBase(m_map.size());*/
            } else if (m_accel == ELightCutAcceleration) {
                VRLLightCut::LCQuery query{ ray, sampler, 0 };
                Li += m_lc->query(scene, sampler, query, nb_BBIntersection, useUniformSampling, useDirectIllum, channel) * m_scale;
                nb_evaluation += query.nb_evaluation;
            } else {
                Log(LogLevel::Error, "query for acceleration is not implemented");
            }
        }
        
        Li /= samples;

        return { nb_evaluation, Li, nb_BBIntersection };
    }

    size_t size() const { return m_map.size(); }

    const VRL &get(int i) const {
        assert(i < size());
        return m_map[i];
    }

    inline void setScaleFactor(Float value) { m_scale = value; }

    const size_t getParticleCount() const { return m_scale; }

protected:
    std::vector<VRL> m_map;
    Float m_scale = 1;
    size_t m_maxSize;
    EVRLAcceleration m_accel;

    VRLLightCut *m_lc;
};

NAMESPACE_END(mitsuba)