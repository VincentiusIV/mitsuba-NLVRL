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

#include "kdtree.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class PhotonMapper : public SamplingIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MTS_IMPORT_TYPES(Emitter)

    struct Photon : PointNode {
        Spectrum spectrum;        
        Normal3f normal;
        Vector3f direction;

        Photon(const Point3f &position, const Normal3f &normal, 
            const Vector3f &direction, const Spectrum &spectrum,
               const int &depth)
            : PointNode() {
            point[0] = position[0];
            point[1] = position[1];
            point[2] = position[2];
            this->spectrum = spectrum;
            this->normal   = normal;
            this->direction = direction;
        }
    };

    class PhotonMap : PointKDTree<Photon> {
    public:
        PhotonMap() { 
            Log(LogLevel::Info, "Constructing PhotonMap...");
        }

        inline void insert(const Photon &photon) { 
               
        }
    };

    PhotonMapper(const Properties &props) : Base(props) {
        m_photonMap = new PhotonMap();
        m_directSamples = props.int_("directSamples", 16);
        m_glossySamples = props.int_("glossySamples", 32);
        m_rrStartDepth  = props.int_("rrStartDepth", 5);
        m_maxDepth      = props.int_("maxDepth", 128);
        m_maxSpecularDepth = props.int_("maxSpecularDepth", 4);
        m_granularity      = props.int_("granularity", 0);
        m_globalPhotons    = props.int_("globalPhotons", 250000);
        m_causticPhotons   = props.int_("causticPhotons", 250000);
        m_volumePhotons    = props.int_("volumePhotons", 250000);
        m_globalLookupRadiusRelative =
            props.float_("globalLookupRadiusRelative", 0.05f);
        m_causticLookupRadiusRelative = props.float_("causticLookupRadiusRelative", 0.0125f);
        m_globalLookupSize = props.int_("globalLookupSize", 120);
        m_causticLookupSize = props.int_("causticLookupSize", 120);
        m_volumeLookupSize  = props.int_("volumeLookupSize", 120);
        m_gatherLocally     = props.bool_("gatherLocally", true);
        m_autoCancelGathering = props.bool_("autoCancelGathering", true);
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


        // 1. For each light source in the scene we create a set of photons 
        //    and divide the overall power of the light source amongst them.
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
                m_photonMap->insert(p);
            }
        }
    }

    MTS_DECLARE_CLASS()
private:
    PhotonMap* m_photonMap;

    int m_directSamples, m_glossySamples, m_rrStartDepth, m_maxDepth,
        m_maxSpecularDepth, m_granularity;
    int m_globalPhotons, m_causticPhotons, m_volumePhotons;
    float m_globalLookupRadiusRelative, m_causticLookupRadiusRelative;
    int m_globalLookupSize, m_causticLookupSize, m_volumeLookupSize;
    /* Should photon gathering steps exclusively run on the local machine? */
    bool m_gatherLocally;
    /* Indicates if the gathering steps should be canceled if not enough photons
     * are generated. */
    bool m_autoCancelGathering;    
};

MTS_IMPLEMENT_CLASS_VARIANT(PhotonMapper, SamplingIntegrator)
MTS_EXPORT_PLUGIN(PhotonMapper, "Photon Mapping integrator");
NAMESPACE_END(mitsuba)
