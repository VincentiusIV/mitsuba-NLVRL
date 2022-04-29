#pragma once

#include <fstream>

NAMESPACE_BEGIN(mitsuba)


#define VRL_DEBUG 0
#define OLD_MITSUBA 1

template <typename Float, typename Spectrum> 
struct VRL {
    MTS_IMPORT_TYPES(PhaseFunctionContext)
    MTS_IMPORT_OBJECT_TYPES()

    VRL() {
        //set VRL as invalid (assume system has quiet NaN)
        origin = Point3f(std::numeric_limits<Float>::quiet_NaN());
        direction = Vector3f(0,0,0);
        length = 0;     
        m_medium = NULL;
        m_longBeams = false;
        m_scatterDepth = -1;
        flux = Spectrum(0.0);
        m_valid = false;
    }

    VRL(const Point3f &p, const Medium *m, const Spectrum &f, int curDepth) :
            origin(p), m_medium(m), flux(f), m_scatterDepth(curDepth) {
        direction = Vector3f(0,0,0);
        length = 0;

        m_longBeams = false;
        m_valid = false;
    }

    std::vector<VRL> dice(const Scene* scene, Sampler* sampler, const Float avgLength) const {
        Ray3f ray(origin, direction, 0.0);
        ray.mint = 0.0;
        ray.maxt = length;

        std::vector<VRL> new_vrls;
        new_vrls.reserve(floor(length / avgLength) + 1);

        // Make the energy equal over the VRL
        auto vrl_trans = [&](Float dist) -> Spectrum {
            if(dist == 0.0) {
                return Spectrum(1.0);
            } else {
                auto _ray = ray;
                _ray.maxt = dist;
                // auto [tr, pdf] = m_medium->eval_tr_and_pdf();
                Mask active = true;
                uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
                uint32_t channel    = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
                return evalMediumTransmittance(m_medium, _ray, sampler, channel, active);
            }
        };

        auto current_distance = 0.0;
        while(current_distance <= length) {
            auto next_length = min(length - current_distance, avgLength);
            // TODO: Make the code more secure by using parameters lists
            VRL new_vrl;
            new_vrl.flux = flux * vrl_trans(current_distance);
            // Regenerate the p1 and p2
            new_vrl.origin = origin + direction * current_distance;
            new_vrl.direction = direction;
            new_vrl.length = next_length;
            // Copy the informations
            new_vrl.m_longBeams = m_longBeams;
            new_vrl.m_medium = m_medium;
            new_vrl.m_scatterDepth = m_scatterDepth;
            new_vrl.m_valid = m_valid;
            new_vrls.emplace_back(std::move(new_vrl));

            current_distance += avgLength;
        }
        return new_vrls;
    }

    int getScatterDepth() const { 
        return m_scatterDepth; 
    }

    Point3f getMiddlePoint() const {
        return origin + direction * length * 0.5;
    }
    void setEndPoint(const Point3f &p) {
        direction = p - origin;
        length = norm(direction);
        direction /= length;
        m_valid = true;
    }

    Point3f getStartPoint() const {
        return origin;
    }

    Point3f getEndPoint() const {
        return origin + direction * length;
    }

    const Medium *getMedium() const { return m_medium; }

    void convertToLong(const Scene *scene) {
        Ray3f r(origin, direction, 0.f);
        SurfaceInteraction3f si = scene->ray_intersect(r, true); 
        if (!si.is_valid()) {
            Log(LogLevel::Error, "Impossible to convert to long vrl...");
        }
        length = si.t;
        m_longBeams = true;
    }

    std::string toString() {
        return "todo";
    }

    struct ClosestPointInfo {
        Float tCam;
        Float tVRL;
    };

    ClosestPointInfo findClosetPoint(const Ray3f &r) const{
        // From http://geomalgorithms.com/a07-_distance.html
        Vector3f u = direction * length;
        Vector3f v = r.d * r.maxt;
        Vector3f w = Vector3f(origin - r.o);

        Float a = enoki::dot(u, u);
        Float b = enoki::dot(u, v); // = nan
        Float c = enoki::dot(v, v); // = inf
        Float d = enoki::dot(u, w); 
        Float e = enoki::dot(v, w);// = -nan

        Float D = a * c - b * b;

        Float sc, sN, sD = D;
        Float tc, tN, tD = D;

        if (D < (Float) 0.0001) {
            sN = (Float) 0.0;
            sD = (Float) 1.0;
            tN = e;
            tD = c;
        } else {
            sN = (b * e - c * d);
            tN = (a * e - b * d);
            if (sN < (Float) 0.0) {        // sc < 0 => the s=0 edge is visible
                sN = (Float) 0.0;
                tN = e;
                tD = c;
            } else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < (Float) 0.0) {            // tc < 0 => the t=0 edge is visible
            tN = (Float) 0.0;
            // recompute sc for this edge
            if (-d < (Float) 0.0)
                sN = (Float) 0.0;
            else if (-d > a)
                sN = sD;
            else {
                sN = -d;
                sD = a;
            }
        } else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
            tN = tD;
            // recompute sc for this edge
            if ((-d + b) < (Float) 0.0)
                sN = (Float) 0;
            else if ((-d + b) > a)
                sN = sD;
            else {
                sN = (-d + b);
                sD = a;
            }
        }

        // finally do the division to get sc and tc
        sc = (std::abs(sN) < (Float) 0.0001 ? (Float) 0.0 : sN / sD);
        tc = (std::abs(tN) < (Float) 0.0001 ? (Float) 0.0 : tN / tD);

#if VRL_DEBUG
        std::ostringstream stome;
        // tc = nan , r.maxt = inf, sc = nan, sN = nan, sD = nan, tN = nan, tD = nan
        stome << "a = " << a << ", b = " << b << ", c = " << c << ", d = " << d << ", e = " << e;
       /* stome << "direction = " << direction << "r = " << r << "tc = " << tc << ", r.maxt = " << r.maxt << ", sc = " << sc << "length = " << length << ", sN = " << sN << ", sD = " << sD
              << ", tN = " << tN
              << ", tD = " << tD;*/
        Log(LogLevel::Info, stome.str().c_str());
#endif

        return ClosestPointInfo {
            tc * r.maxt,
            sc * length
        };
    }

    struct SamplingInfo {
        Point3f pCam;
        Float tCam;
        Point3f pVRL;
        Float tVRL;
        Float invPDF;

        static SamplingInfo invalid() {
            return SamplingInfo {
                Point3f(0.0),
                0,
                Point3f(0.0),
                0,
                0
            };
        }
    };

    SamplingInfo sampleMC(const Ray3f& ray, Sampler* sampler) const{
        Float tCam = sampler->next_1d() * ray.maxt;
        Float tVRL =  sampler->next_1d() * length;

        return SamplingInfo {
            ray.o + ray.d * tCam,
            tCam,
            origin + direction * tVRL,
            tVRL,
            length * ray.maxt
        };
    }


#define USE_PEAK_SAMPLING 0
#define USE_ANISOTROPIC_SAMPLING 1
    SamplingInfo samplingVRL(const Scene* scene, const Ray3f& ray, Sampler* sampler, bool uniformSampling, UInt32 channel) const {
        if (uniformSampling) {
            return sampleMC(ray, sampler);
        } else {
            auto closest_point = findClosetPoint(ray);
            auto h = [&closest_point, this, &ray]() -> Float {
                Point3f vh = origin + direction * closest_point.tVRL;
                Point3f uh = ray.o + ray.d * closest_point.tCam;
                return norm(uh - vh);
            }();

            // prevent degenerative cases when VRL/sensor are very close
            if (h == 0.0) return SamplingInfo::invalid();
            h = sqrt(h);

            // Code from sampling VRL points
            // Section 4.1
            Float v0Hat = -closest_point.tVRL;
            Float v1Hat = length + v0Hat;

            // Compute the sin(theta) using cross product
            Float sinTheta = norm(cross(direction, ray.d));
            if(sinTheta == 0) {
                Log(LogLevel::Warn, "Parallel vectors");
                return SamplingInfo::invalid();
            }
            assert(sinTheta > 0);
            sinTheta = sqrt(sinTheta);

            // Alternative code for compute sin(theta) much slower
            // TODO: Direct formula for computing sin theta
//                Float tmpDot = dot(direction, ray.d);
//                Float dotProd = (Float) std::acos(std::max((Float) -1.0, std::min(tmpDot, (Float) 1.0)));
//                Float sinTheta = std::sin(dotProd);

            // Sampling functions
            struct VRLSampling {
                Float vHat;
                Float invPDF;
            };
            auto inverseCDF_A = [](Float r, Float v0Hat, Float v1Hat, Float h, Float sinTheta) -> VRLSampling {
                auto A = [](Float x, Float h, Float sinTheta) -> Float {
                    // TODO: Check if a better and more efficient alternative is available 
                    //  for computing the asinh function
                    auto asinh = [](const Float value) -> Float {
                        return std::log(value + std::sqrt(value * value + 1));
                    };
                    return asinh((x / h) * sinTheta);
                };
                // Equation 13
                Float a0 = A(v0Hat, h, sinTheta);
                Float a1 = A(v1Hat, h, sinTheta);
                Float v = (h * std::sinh(lerp(r, a0, a1))) / sinTheta;

                // Equation 10 and 11
                Float invPDF = (a1 - a0) * sqrt(h*h + v*v*sinTheta*sinTheta);
                invPDF /= sinTheta;
                return VRLSampling {
                        v,
                        invPDF
                };
            };

            auto sampling_vrl = inverseCDF_A(sampler->next_1d(), v0Hat, v1Hat, h, sinTheta);
            // Rechange the variable from vHat to V;
            Point3f pVRL = origin + direction * (sampling_vrl.vHat + closest_point.tVRL);
            // vHat = nan, tVRL = nan

#if VRL_DEBUG
            std::ostringstream stome;
            stome << "o = " << origin << ", d = " << direction << ", pVRL = " << pVRL << ", vHat = " << sampling_vrl.vHat << ", tVRL = " << closest_point.tVRL;
            Log(LogLevel::Info, stome.str().c_str());
#endif
            // sample point on camera ray using Kulla and al. : section 4.1
#if USE_ANISOTROPIC_SAMPLING
            const PhaseFunction *pf = m_medium->phase_function();
            if (has_flag(pf->flags(true), PhaseFunctionFlags::Isotropic)) {
#else
            if(true) {
#endif
                Float uHat = dot(ray.d, (pVRL-ray.o));
                Float u0Hat = -uHat;
                Float u1Hat = ray.maxt + u0Hat;

                struct RaySampling {
                    Float uHat;
                    Float invPDF;
                };
                auto inverseCDF_B = [](Float eps, Float u0, Float u1, Float h) -> RaySampling {
                    auto B = [](Float x, Float h) -> Float { return atan(x / h); };
                    Float thetaA = B(u0, h);
                    Float thetaB = B(u1, h);
                    Float uHat = h * tan(lerp(eps, thetaA, thetaB));
                    return RaySampling{
                            uHat,
                            (thetaB - thetaA) * (h*h + uHat * uHat) / h
                    };
                };
                Float hPoint = norm((ray.o + ray.d * uHat) - (pVRL));
                auto sampling_ray = inverseCDF_B(sampler->next_1d(), u0Hat, u1Hat, hPoint);
                Point3f pCam = ray.o + ray.d * (sampling_ray.uHat - u0Hat);

                return SamplingInfo{
                        pCam,
                        (sampling_ray.uHat - u0Hat),
                        pVRL,
                        (sampling_vrl.vHat + closest_point.tVRL),
                        sampling_vrl.invPDF * sampling_ray.invPDF
                };
            } else {
                // Anisotropic phase function
                // Note this code is not "battle" tested.

#define CDF_LENGHT 10 
                // Precompute values (Appendix 1)
                Vector3f a = (ray.o - pVRL);
                Float distVRLtoOrigin = norm(a);
                a /= distVRLtoOrigin;
                Vector3f b = normalize(ray(ray.maxt) - pVRL);
                Vector3f c = cross(a, b);

                // Initialize the thetas
                // These will be used to sample the interval
                Float theta[CDF_LENGHT];

                // This part initialize the theta in the kulla angluar space
                Float uHat = dot(ray.d, (pVRL-ray.o));
                Float u0Hat = -uHat;
                Float u1Hat = ray.maxt + u0Hat;
                Float hPoint = norm((ray.o + ray.d * uHat) - pVRL);
                theta[0] = atan(u0Hat/hPoint) + M_PI_2;
                theta[CDF_LENGHT - 1] = atan(u1Hat/hPoint) + M_PI_2;

                // Get the peak of phase function (negative or positive
                Float peak = acos(dot(a, normalize(cross(cross(c, this->direction), c))));
                if (peak < theta[0] || peak > theta[CDF_LENGHT - 1]) {
                    peak = acos(dot(a, normalize(cross(cross(c, -this->direction), c))));
                }

                // Fill the theta table
                // Note that this tetha can be used directly to compute the phase function
                // From the VRL phase function
#if USE_PEAK_SAMPLING
                if (peak > theta[0] && peak < theta[CDF_LENGHT - 1]) {
#else
                if(false) {
#endif
                    // The peak is found, ensure to have a sample on it
                    // The rest, distribute using the cosine weighted strategy (Equation 19)
                    int index = std::floor(((peak - theta[0]) / (theta[CDF_LENGHT - 1] - theta[0]))
                                           * Float(CDF_LENGHT - 1) - 0.5f);
                    theta[index] = peak;
                    if(index == 0 || index == CDF_LENGHT - 1) {
                        // In this case we use the normal approach
                        for (int i = 1; i < CDF_LENGHT - 1; i++) {
                            // Cosine weighted spaced
                            theta[i] = ((theta[CDF_LENGHT - 1] - theta[0]) * 0.5) *
                                       (1 - std::cos(M_PI * (i / Float(CDF_LENGHT - 1)))) + theta[0];
                        }
                    } else {
                        // 1 .. index peak
                        for (int i = 1; i < index; i++) {
                            // [0,2]
                            Float cos_factor = 1 - std::cos(M_PI * (i / Float(index)));
                            theta[i] = (peak - theta[0]) * 0.5 * cos_factor + theta[0];
                        }
                        // index peak ... M
                        // TODO: Checkthe denominator CDF_LENGHT - index
                        for (int i = index + 1; i < CDF_LENGHT - 1; i++) {
                            Float cos_factor = 1 - std::cos(M_PI * (i / Float(CDF_LENGHT - 1 - index)));
                            theta[i] = (theta[CDF_LENGHT - 1] - peak) * 0.5 * cos_factor + peak;
                        }
                    }
                } else {
                    // The peak is not inside the interval. Use cosine wrapped strategy
                    // Equation 19
                    for (int i = 1; i < CDF_LENGHT - 1; i++) {
                        // Cosine weighted spaced
                        theta[i] = ((theta[CDF_LENGHT - 1] - theta[0]) * 0.5) *
                                   (1 - std::cos(M_PI * (i / Float(CDF_LENGHT - 1)))) + theta[0];
                    }
                }

                //find the position based on the angle
                Float tRay[CDF_LENGHT]; //< The distance on the camera ray
                Float phaseFunctions[CDF_LENGHT]; //< The product of phase functions
                Float thetaRayVPL = acos(dot(-a, ray.d));
                Mask active = true;
                for (int i = 0; i < CDF_LENGHT; i++) {

                    // Uses kulla equation to get the distance on the camera ray
                    tRay[i] = (hPoint * tan(theta[i] - M_PI_2)) - u0Hat;

                    Vector dir = normalize(pVRL - ray(tRay[i]));
                    /*const PhaseFunction *pf = m_medium->phase_function();
                    MediumSamplingRecord mRec1;
                    PhaseFunctionSamplingRecord psr1(mRec1, -this->direction, -dir);
                    MediumSamplingRecord mRec2;
                    PhaseFunctionSamplingRecord psr2(mRec2, -ray.d, dir);

                    Float vrlPF = pf->eval(psr1);
                    Float rayPF = pf->eval(psr2);
                    phaseFunctions[i] = vrlPF * rayPF;*/

                    const PhaseFunction *pf = m_medium->phase_function();
                    Float vrlPF             = 1.0;
                    Float rayPF             = 1.0;

                    Spectrum sigmaSRay, sigmaSVRL;
                    PhaseFunctionContext phase_ctx(sampler);
                    
                    Ray3f mediumVRL(origin, direction, 0);

                    SurfaceInteraction3f vrl_si  = scene->ray_intersect(mediumVRL);
                    MediumInteraction3f vrl_mi   = m_medium->sample_interaction(mediumVRL, sampler->next_1d(), channel, active);
                    auto [vrl_tr, vrl_pdf]       = m_medium->eval_tr_and_pdf(vrl_mi, vrl_si, active);
                    sigmaSVRL                    = vrl_mi.sigma_s;
                    //auto [vrl_wo, vrl_phase_pdf] = pf->sample(phase_ctx, vrl_mi, sampler->next_2d(active), active);
                    vrlPF                        = pf->eval(phase_ctx, vrl_mi, -dir);

                    SurfaceInteraction3f ray_si  = scene->ray_intersect(ray);
                    MediumInteraction3f ray_mi   = m_medium->sample_interaction(ray, sampler->next_1d(), channel, active);
                    auto [ray_tr, ray_pdf]       = m_medium->eval_tr_and_pdf(ray_mi, ray_si, active);
                    sigmaSRay                    = ray_mi.sigma_s;
                    //auto [ray_wo, ray_phase_pdf] = pf->sample(phase_ctx, ray_mi, sampler->next_2d(active), active);
                    rayPF                        = pf->eval(phase_ctx, ray_mi, dir);
                    
                    phaseFunctions[i] = vrlPF * rayPF;
                }

                // This code compute the integral piece wise
                // Using the triangle deduction
                Float icdf[CDF_LENGHT];
                icdf[0] = 0;
                for (int i = 0; i < CDF_LENGHT - 1; i++) {
                    Float min = std::min(phaseFunctions[i], phaseFunctions[i + 1]);
                    Float max = std::max(phaseFunctions[i], phaseFunctions[i + 1]);

                    // Kulla, we work inside the theta space
                    Float delta_domain = theta[i+1] - theta[i];

                    // Compute the area and accumulate it
                    Float area = delta_domain*min;
                    Float diff = (max - min)*delta_domain;
                    area += 0.5f * diff;
                    icdf[i+1] = icdf[i] + area;
                }

                // Normalize the pdf and CDF
                // TODO: Only need to compute the sum of the element
                // the normalization here and constructing the CDF
                // is just wastefull here
                Float mul = 1.f / icdf[CDF_LENGHT-1];
                for (int i = 0; i < CDF_LENGHT; i++) {
                    phaseFunctions[i] *= mul;
                    icdf[i] *= mul;
                }

                Float t = sampler->next_1d();
                int index = 0;
                for (;index < CDF_LENGHT - 1; index++) {
                    if (t > icdf[index] && t < icdf[index + 1]) {
                        break;
                    }
                }
                // TODO: not sure that linear interpolation is a good choice here
                // Renormlize the t to [0,1] inside the interval
                t = (t - icdf[index]) / (icdf[index + 1] - icdf[index]);

                Float thetaInter = t * (theta[index + 1] - theta[index]) + theta[index];
                Float tCam = hPoint * tan(thetaInter - M_PI_2);

                Float probCDF = t * (phaseFunctions[index + 1] - phaseFunctions[index]) + phaseFunctions[index];
                probCDF *= hPoint / (hPoint*hPoint + tCam*tCam); 
                
                tCam -= u0Hat;

                return SamplingInfo {
                        ray.o + ray.d * tCam,
                        tCam,
                        pVRL,
                        (sampling_vrl.vHat + closest_point.tVRL),
                        sampling_vrl.invPDF / probCDF
                };
            }
        }
    }

    Spectrum evalTransmittance(const Scene *scene, const Point3f &p1, bool p1OnSurface, const Point3f &p2, bool p2OnSurface, Float time, const Medium *medium, int &interactions, Sampler *sampler,
                               UInt32 channel) const {
        Vector3f d        = p2 - p1;
        Float remaining = norm(d);
        d /= remaining;

        Float lengthFactor = p2OnSurface ? (1 - math::ShadowEpsilon<Float>) : 1;
        Ray3f ray(p1, d, time);
        ray.mint = p1OnSurface ? math::Epsilon<Float> : 0;
        ray.maxt = remaining * lengthFactor;

        Spectrum transmittance(1.0f);
        int maxInteractions = interactions;
        interactions        = 0;

        SurfaceInteraction3f si;

        Mask active_surface = true;

        while (remaining > 0) {
            si = scene->ray_intersect(ray); // , its.t, its.shape, its.geoFrame.n, its.uv

            active_surface &= si.is_valid();
            if (none(active_surface))
                break;
            if (any_or<true>(active_surface) && (interactions == maxInteractions || !(has_flag(si.bsdf()->flags(), BSDFFlags::Null)))) {
                /* Encountered an occluder -- zero transmittance. */
                return Spectrum(0.0f);
            }

            if (medium) {
                /*mi             = medium->sample_interaction(ray, sampler->next_1d(), channel, active_surface);
                auto [tr, pdf] = medium->eval_tr_and_pdf(mi, si, active_surface);
                transmittance *= tr;*/

                Ray3f r(ray);
                r.mint = 0;
                r.maxt         = enoki::min(si.t, remaining);
                transmittance *= evalMediumTransmittance(medium, r, sampler, channel, active_surface);                
            }

            if (transmittance == Spectrum(0.0f))
                break;

            const BSDF *bsdf = si.bsdf();

            Vector wo = si.to_local(ray.d);
            BSDFContext bCtx;
            bCtx.type_mask = bCtx.type_mask & BSDFFlags::Null;
            transmittance *= bsdf->eval(bCtx, si, wo);

            if (si.is_medium_transition()) {
                if (medium != si.target_medium(-d)) {
                    //++mediumInconsistencies;
                    //Log(LogLevel::Warn, "medium inconsistency?");
                    return Spectrum(0.0f);
                }
                medium = si.target_medium(d);
            }

            if (++interactions > 100) { /// Just a precaution..
                Log(LogLevel::Warn, "evalTransmittance(): round-off error issues?");
                break;
            }

            ray.o += ray(si.t);
            remaining -= si.t;
            ray.maxt = remaining * lengthFactor;
            ray.mint = math::Epsilon<Float>;
        }

        return transmittance;
    }

    static Spectrum evalMediumTransmittance(const Medium *medium, const Ray3f &ray, Sampler *sampler, UInt32 channel, Mask active) {
        auto [hit, mint, maxt] = medium->intersect_aabb(ray);
        hit &= (enoki::isfinite(mint) || enoki::isfinite(maxt));
        active &= hit;
        masked(mint, !active) = 0.f;
        masked(maxt, !active) = math::Infinity<Float>;
        if (none(hit))
            return Spectrum(1.0f);
        mint = std::max(mint, ray.mint);
        maxt = std::min(maxt, ray.maxt);

        int nSamples = 2; /// XXX make configurable
        Float result = 0;

        for (int i = 0; i < nSamples; ++i) {
            Float t = mint;
            Ray3f r(ray);
            while (true) {
                MediumInteraction3f mi = medium->sample_interaction(r, sampler->next_1d(), channel, active);
                t += mi.t;
                if (t >= maxt || !mi.is_valid()) {
                    result += 1;
                    break;
                }

                
                auto combined_extinction = medium->get_combined_extinction(mi, active);
                Float m                    = mi.sigma_t[0] * combined_extinction[0];
                masked(m, eq(channel, 1u)) = mi.sigma_t[1] * combined_extinction[1];
                masked(m, eq(channel, 2u)) = mi.sigma_t[2] * combined_extinction[2];

                r.o = ray(t);

                if (m > sampler->next_1d())
                    break;
            }
        }
        return Spectrum(result / nSamples);
    }

    Spectrum getContrib(
            const Scene* scene,
            const bool uniformSampling,
            const Ray3f& ray,
            Float lengthOfRay,
            Sampler* sampler, UInt32 channel) const {
        //
        //sample VRL & camera ray uniformly
        //
        auto sampling = samplingVRL(scene, ray, sampler, uniformSampling, channel);

        /*std::ostringstream stome;
        stome << "pVRL = " << sampling.pVRL << ", pCam = " << sampling.pCam;
        Log(LogLevel::Info, stome.str().c_str());*/

        // Check the visibility of the two sampled points
        Vector3f dir = sampling.pVRL - sampling.pCam;
        Float lengthPtoP = norm(dir);
        if (lengthPtoP == 0) {
            Log(LogLevel::Warn, "0 distance between VRL and Cam ray");
            return Spectrum(0.0);
        }
        dir /= lengthPtoP;

        Mask active = true;
        Ray3f mediumPtoP(sampling.pCam, dir, 0);
        mediumPtoP.mint = 0;
        mediumPtoP.maxt = lengthPtoP; 

        Ray3f mediumRay(ray.o, ray.d, 0);
        mediumRay.mint   = 0;
        mediumRay.maxt   = sampling.tCam;

        int interactions = 100;

        SurfaceInteraction3f si = scene->ray_intersect(mediumPtoP);
        MediumInteraction3f mi  = m_medium->sample_interaction(mediumPtoP, sampler->next_1d(), channel, active);

        if (!mi.is_valid())
            return Spectrum(0.0);
        
#if OLD_MITSUBA
        Spectrum vrlToRayTrans= evalTransmittance(scene, sampling.pCam, false, sampling.pVRL, false, 0, m_medium, interactions, sampler, channel);
        Spectrum rayTrans = evalMediumTransmittance(m_medium, mediumRay, sampler, channel, active);
#else
        auto [vrlToRay_tr, vrlToRay_pdf] = m_medium->eval_tr_and_pdf(mi, si, active);
        Spectrum vrlToRayTrans = vrlToRay_tr;

        SurfaceInteraction3f mediumRay_si = scene->ray_intersect(mediumRay);
        MediumInteraction3f mediumRay_mi = m_medium->sample_interaction(mediumRay, sampler->next_1d(), channel, active);
        auto [mediumRay_tr, mediumRay_pdf] = m_medium->eval_tr_and_pdf(mi, si, active);
        Spectrum rayTrans = mediumRay_tr;

        if (!mediumRay_mi.is_valid())
            return Spectrum(0.0);
#endif
        auto vrlTrans = [&]() -> Spectrum {
            if(m_longBeams) {
                Ray3f mediumVRL(origin, direction, 0);
                mediumVRL.mint = 0;
                mediumVRL.maxt = sampling.tVRL;

#if OLD_MITSUBA
                
                return evalMediumTransmittance(m_medium, mediumVRL, sampler, channel, active);
#else
                SurfaceInteraction3f medium_si = scene->ray_intersect(mediumVRL);
                MediumInteraction3f mi         = m_medium->sample_interaction(mediumVRL, sampler->next_1d(), channel, active);

                if (!mediumRay_mi.is_valid())
                    return Spectrum(1.f);
                auto [tr, pdf]                 = m_medium->eval_tr_and_pdf(mi, medium_si, active);
                return tr;
#endif
            } else {
                return Spectrum(1.f);
            }
        }();

        Float fallOff = 1.0f / (lengthPtoP * lengthPtoP);


    #if VRL_DEBUG
        std::ostringstream streammm;
        streammm << "fallOff = " << fallOff << ", lengthPtoP = " << lengthPtoP;
        Log(LogLevel::Info, streammm.str().c_str());
    #endif 

        MediumInteraction3f mi1, mi2;

        const PhaseFunction *pf = m_medium->phase_function();
        PhaseFunctionContext pCtx1(sampler), pCtx2(sampler);
        Float vrlPF = 1.0; // wi = -direction, wo = -dir
        Float rayPF = 1.0; // wi = -ray.d, wo = dir

        Spectrum sigmaSRay, sigmaSVRL;
        if (m_medium->is_homogeneous()) {
            auto [sigma_s, sigma_n, sigma_t] = m_medium->get_scattering_coefficients(mi);
            sigmaSRay = sigma_s;
            sigmaSVRL = sigma_s;
        } else {
            Ray3f mediumVRL(origin, direction, 0);
            mediumVRL.mint = 0;
            mediumVRL.maxt = sampling.tVRL;
            SurfaceInteraction3f vrl_si = scene->ray_intersect(mediumVRL);
            MediumInteraction3f vrl_mi = m_medium->sample_interaction(mediumVRL, sampler->next_1d(), channel, active);
            auto [vrl_tr, vrl_pdf] = m_medium->eval_tr_and_pdf(vrl_mi, vrl_si, active);
            sigmaSVRL = vrl_mi.sigma_s;
            vrlPF = pf->eval(pCtx1, vrl_mi, -dir);

            SurfaceInteraction3f ray_si = scene->ray_intersect(mediumRay);
            MediumInteraction3f ray_mi = m_medium->sample_interaction(mediumRay, sampler->next_1d(), channel, active);
            auto [ray_tr, ray_pdf]     = m_medium->eval_tr_and_pdf(ray_mi, ray_si, active);
            sigmaSRay                  = ray_mi.sigma_s;
            rayPF = pf->eval(pCtx2, ray_mi, dir);
        }
#if VRL_DEBUG
        std::ostringstream stream;
        stream << "Contrib VRL = [flux:" << flux << ", fallOff:" << fallOff << ", vrlTrans:" << vrlTrans << ", vrlPF:" << vrlPF << ", rayPF:" << rayPF << ", rayTrans:" << rayTrans
               << ", vrlToRayTrans:" << vrlToRayTrans << ", sigmaSRay:" << sigmaSRay << ", sigmaSVRL:" << sigmaSVRL << ", invPDF:" << sampling.invPDF;
        std::string str = stream.str();
        Log(LogLevel::Info, str.c_str());
#endif

        return flux
               * fallOff // = nan
               * vrlTrans // 1.0 if short beams
               * vrlPF //Fs(theta u0)
               * rayPF //Fs(theta uv)
               * rayTrans // = 0
               * vrlToRayTrans // = 0
               * sigmaSRay 
               * sigmaSVRL
               * sampling.invPDF; // = nan
    }

public:
    // Reduce the burden for the implementation
    Point3f origin;
    Vector3f direction;
    Float length;
    Spectrum flux;

protected:
    bool m_longBeams;
    const Medium *m_medium;
    int m_scatterDepth;
    bool m_valid;
};

template <typename Float, typename Spectrum> std::ostream &operator<<(std::ostream &os, const VRL<Float, Spectrum> &vrl) {
    os << "VRL";
    os << "[" << std::endl << "  o = " << vrl.origin << "," << std::endl << "  d = " << vrl.direction << std::endl <<
    "," << std::endl << " l = " << vrl.length << ", " << std::endl << " flux = " << vrl.flux << "]";
    return os;
}

NAMESPACE_END(mituba)
