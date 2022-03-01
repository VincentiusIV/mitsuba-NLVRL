#pragma once

#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(Emitter)

    struct Photon {
        Point3f position;
        Normal3f normal;
        Vector3f direction;
        Spectrum spectrum;
        int depth;

        Photon(const Point3f &p, const Normal3f &n, const Vector3f &d,
               const Spectrum &P, const int _depth) {
            position = p;
            normal   = n;
            direction = d;
            spectrum  = P;
            depth     = _depth;
        }
    };


    class PhotonMap {
    public:
        PhotonMap() { 
            Log(LogLevel::Info, "Constructing PhotonMap...");
        }

        inline void pushBack(const Photon &photon) {
            
        }
    };


    PhotonMapper(const Properties &props) : Base(props) {
        m_photonMap = new PhotonMap();
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium * medium,
                                     Float * aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        static bool m_isPreProcessed = false;
        if (!m_isPreProcessed) {
            m_isPreProcessed = true;
            preProcess(scene);
        }

        Spectrum result(1.f);

        // 1. Traverse KD-tree to find nearby photons.

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        

        return { 
            result, si.is_valid() 
        };
    }

    void preProcess(const Scene *scene) const {
        Log(LogLevel::Info, "Pre Processing Photon Map...");

        const int n = 100000;

        // 1. For each light...
        host_vector<ref<Emitter>, Float> emitters = scene->emitters();
        std::string log = "Scene emitters: " + std::to_string(emitters.size());
        Log(LogLevel::Info, log.c_str());  
        for (const auto &emitter : emitters) {
            // Propagate n photons with intensity 1 in random directions
            for (int i = 0; i < n; i++) {
                int depth(1);
                Spectrum photonIntensity(1.0f);
                Point3f emitterPosition;
                Vector3f photonDirection;
                Float time(.0f);        
                
                // 2. Trace photon until it collides with an object.
                Ray3f photonRay(emitterPosition, photonDirection, time);
                RayDifferential3f  photonRayDiff(photonRay);
                SurfaceInteraction3f si = scene->ray_intersect(photonRayDiff, true);
                Normal3f normal = si.n;
                Point3f interactionPoint = si.p;

                // 3. Russian-Roulette whether photon is reflected or absorbed


                // 4. If absorbed, store photon in photonmap
                Photon p(interactionPoint, normal, photonDirection, photonIntensity, depth);
                m_photonMap->pushBack(p);
            }
        }
    }

    MTS_DECLARE_CLASS()
private:
    PhotonMap* m_photonMap;
};

MTS_IMPLEMENT_CLASS_VARIANT(PhotonMapper, SamplingIntegrator)
MTS_EXPORT_PLUGIN(PhotonMapper, "Photon Mapping integrator");
NAMESPACE_END(mitsuba)
