#pragma once

#include <mitsuba/core/vector.h>
#include <mitsuba/core/ray.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Generic n-dimensional bounding box data structure
 *
 * Maintains a minimum and maximum position along each dimension and provides
 * various convenience functions for querying and modifying them.
 *
 * This class is parameterized by the underlying point data structure,
 * which permits the use of different scalar types and dimensionalities, e.g.
 * \code
 * BoundingBox<Point3i> integer_bbox(Point3i(0, 1, 3), Point3i(4, 5, 6));
 * BoundingBox<Point2d> double_bbox(Point2d(0.0, 1.0), Point2d(4.0, 5.0));
 * \endcode
 *
 * \tparam T The underlying point data type (e.g. \c Point2d)
 */
template <typename Point_> struct BoundingBox {
    static constexpr size_t Dimension = array_size_v<Point_>;
    using Point                       = Point_;
    using Value                       = value_t<Point>;
    using Scalar                      = scalar_t<Point>;
    using Vector                      = typename Point::Vector;
    using UInt32                      = uint32_array_t<Value>;
    using Mask                        = mask_t<Value>;

    /**
     * \brief Create a new invalid bounding box
     *
     * Initializes the components of the minimum and maximum position to
     * \f$\infty\f$ and \f$-\infty\f$, respectively.
     */
    BoundingBox() { reset(); }

    /// Create a collapsed bounding box from a single point
    BoundingBox(const Point &p)
        : min(p), max(p) { }

    /// Create a bounding box from two positions
    BoundingBox(const Point &min, const Point &max)
        : min(min), max(max) { }

    /// Create a bounding box from a smaller type (e.g. vectorized from scalar).
    template <typename PointT>
    BoundingBox(const BoundingBox<PointT> &other)
        : min(other.min), max(other.max) { }

    /// Test for equality against another bounding box
    bool operator==(const BoundingBox &bbox) const {
        return all_nested(eq(min, bbox.min) && eq(max, bbox.max));
    }

    /// Test for inequality against another bounding box
    bool operator!=(const BoundingBox &bbox) const {
        return any_nested(neq(min, bbox.min) || neq(max, bbox.max));
    }

    /**
     * \brief Check whether this is a valid bounding box
     *
     * A bounding box \c bbox is considered to be valid when
     * \code
     * bbox.min[i] <= bbox.max[i]
     * \endcode
     * holds for each component \c i.
     */
    Mask valid() const {
        return all(max >= min);
    }

    /// Check whether this bounding box has collapsed to a point, line, or plane
    Mask collapsed() const {
        return any(eq(min, max));
    }

    /// Return the dimension index with the index associated side length
    UInt32 major_axis() const {
        Vector d = max - min;
        UInt32 index = 0;
        Value value = d[0];

        for (uint32_t i = 1; i < Dimension; ++i) {
            auto mask = d[i] > value;
            masked(index, mask) = i;
            masked(value, mask) = d[i];
        }

        return index;
    }

    /// Return the dimension index with the shortest associated side length
    UInt32 minor_axis() const {
        Vector d = max - min;
        UInt32 index(0);
        Value value = d[0];

        for (uint32_t i = 1; i < Dimension; ++i) {
            Mask mask = d[i] < value;
            masked(index, mask) = i;
            masked(value, mask) = d[i];
        }

        return index;
    }

    /// Return the center point
    Point center() const {
        return (max + min) * Scalar(.5f);
    }

    /**
     * \brief Calculate the bounding box extents
     * \return <tt>max - min</tt>
     */
    Vector extents() const { return max - min; }

    /// Return the position of a bounding box corner
    Point corner(size_t index) const {
        Point result;
        for (uint32_t i = 0; i < uint32_t(Dimension); ++i)
            result[i] = ((uint32_t) index & (1 << i)) ? max[i] : min[i];
        return result;
    }

    /// Calculate the n-dimensional volume of the bounding box
    Value volume() const { return hprod(max - min); }

    /// Calculate the 2-dimensional surface area of a 3D bounding box
    Value surface_area() const {
        if constexpr (Dimension == 3) {
            /// Fast path for n = 3
            Vector d = max - min;
            return hsum(enoki::shuffle<1, 2, 0>(d) * d) * Scalar(2);
        } else {
            /// Generic case
            Vector d = max - min;

            Value result = Scalar(0);
            for (size_t i = 0; i < Dimension; ++i) {
                Value term = Scalar(1);
                for (size_t j = 0; j < Dimension; ++j) {
                    if (i == j)
                        continue;
                    term *= d[j];
                }
                result += term;
            }
            return result * Scalar(2);
        }
    }

    /**
     * \brief Check whether a point lies \a on or \a inside the bounding box
     *
     * \param p The point to be tested
     *
     * \tparam Strict Set this parameter to \c true if the bounding
     *                box boundary should be excluded in the test
     *
     * \remark In the Python bindings, the 'Strict' argument is a normal
     *         function parameter with default value \c False.
     */
    template <bool Strict = false, typename T, typename Result = mask_t<expr_t<T, Value>>>
    Result contains(const mitsuba::Point<T, Point::Size> &p) const {
        if constexpr (Strict)
            return all((p > min) && (p < max));
        else
            return all((p >= min) && (p <= max));
    }

    /**
     * \brief Check whether a specified bounding box lies \a on or \a within
     * the current bounding box
     *
     * Note that by definition, an 'invalid' bounding box (where min=\f$\infty\f$
     * and max=\f$-\infty\f$) does not cover any space. Hence, this method will always
     * return \a true when given such an argument.
     *
     * \tparam Strict Set this parameter to \c true if the bounding
     *                box boundary should be excluded in the test
     *
     * \remark In the Python bindings, the 'Strict' argument is a normal
     *         function parameter with default value \c False.
     */
    template <bool Strict = false, typename T, typename Result = mask_t<expr_t<T, Value>>>
    Result contains(const BoundingBox<mitsuba::Point<T, Point::Size>> &bbox) const {
        if constexpr (Strict)
            return all((bbox.min > min) && (bbox.max < max));
        else
            return all((bbox.min >= min) && (bbox.max <= max));
    }

    /**
     * \brief Check two axis-aligned bounding boxes for possible overlap.
     *
     * \param Strict Set this parameter to \c true if the bounding
     *               box boundary should be excluded in the test
     *
     * \remark In the Python bindings, the 'Strict' argument is a normal
     *         function parameter with default value \c False.
     *
     * \return \c true If overlap was detected.
     */
    template <bool Strict = false, typename T, typename Result = mask_t<expr_t<T, Value>>>
    Result overlaps(const BoundingBox<mitsuba::Point<T, Point::Size>> &bbox) const {
        if constexpr (Strict)
            return all((bbox.min < max) && (bbox.max > min));
        else
            return all((bbox.min <= max) && (bbox.max >= min));
    }

    /**
     * \brief Calculate the shortest squared distance between
     * the axis-aligned bounding box and the point \c p.
     */
    template <typename T, typename Result = expr_t<T, Value>>
    Result squared_distance(const mitsuba::Point<T, Point::Size> &p) const {
        return squared_norm(((min - p) & (p < min)) + ((p - max) & (p > max)));
    }

    /**
     * \brief Calculate the shortest squared distance between
     * the axis-aligned bounding box and \c bbox.
     */
    template <typename T, typename Result = expr_t<T, Value>>
    Result squared_distance(const BoundingBox<mitsuba::Point<T, Point::Size>> &bbox) const {
        return squared_norm(((min - bbox.max) & (bbox.max < min)) +
                            ((bbox.min - max) & (bbox.min > max)));
    }

    /**
     * \brief Calculate the shortest distance between
     * the axis-aligned bounding box and the point \c p.
     */
    template <typename T, typename Result = expr_t<T, Value>>
    Result distance(const mitsuba::Point<T, Point::Size> &p) const {
        return enoki::sqrt(squared_distance(p));
    }

    /**
     * \brief Calculate the shortest distance between
     * the axis-aligned bounding box and \c bbox.
     */
    template <typename T, typename Result = expr_t<T, Value>>
    Result distance(const BoundingBox<mitsuba::Point<T, Point::Size>> &bbox) const {
        return enoki::sqrt(squared_distance(bbox));
    }

    /**
     * \brief Mark the bounding box as invalid.
     *
     * This operation sets the components of the minimum
     * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
     * respectively.
     */
    void reset() {
        min =  std::numeric_limits<Value>::infinity();
        max = -std::numeric_limits<Value>::infinity();
    }

    /// Clip this bounding box to another bounding box
    template <typename T>
    void clip(const BoundingBox<mitsuba::Point<T, Point::Size>> &bbox) {
        min = enoki::max(min, bbox.min);
        max = enoki::min(max, bbox.max);
    }

    /// Expand the bounding box to contain another point
    template <typename T>
    void expand(const mitsuba::Point<T, Point::Size> &p) {
        min = enoki::min(min, p);
        max = enoki::max(max, p);
    }

    /// Expand the bounding box to contain another bounding box
    template <typename T>
    void expand(const BoundingBox<mitsuba::Point<T, Point::Size>> &bbox) {
        min = enoki::min(min, bbox.min);
        max = enoki::max(max, bbox.max);
    }

    /// Merge two bounding boxes
    static BoundingBox merge(const BoundingBox &bbox1, const BoundingBox &bbox2) {
        return BoundingBox(
            enoki::min(bbox1.min, bbox2.min),
            enoki::max(bbox1.max, bbox2.max)
        );
    }

    /**
     * \brief Check if a ray intersects a bounding box
     *
     * Note that this function ignores the <tt>(mint, maxt)</tt> interval
     * associated with the ray.
     */
    template <typename Ray>
    MTS_INLINE auto ray_intersect(const Ray &ray) const {
        using Float  = typename Ray::Float;
        using Vector = typename Ray::Vector;

        /* First, ensure that the ray either has a nonzero slope on each axis,
           or that its origin on a zero-valued axis is within the box bounds */
        auto active = all(neq(ray.d, zero<Vector>()) || ((ray.o > min) || (ray.o < max)));

        // Compute intersection intervals for each axis
        Vector t1 = (min - ray.o) * ray.d_rcp,
               t2 = (max - ray.o) * ray.d_rcp;

        // Ensure proper ordering
        Vector t1p = enoki::min(t1, t2),
               t2p = enoki::max(t1, t2);

        // Intersect intervals
        Float mint = hmax(t1p),
              maxt = hmin(t2p);

        active = active && maxt >= mint;

        return std::make_tuple(active, mint, maxt);
    }

    /// Create a bounding sphere, which contains the axis-aligned box
    template <typename Result = BoundingSphere<Point>>
    Result bounding_sphere() const {
        Point c = center();
        return { c, norm(c - max) };
    }


    template <typename Ray> MTS_INLINE auto getMinDistanceSqr(const Ray &ray) const {
        using Float  = typename Ray::Float;
        using Vector = typename Ray::Vector;
    // Utility code to compute the min distance AABB
    class MinDistanceAABB {
    public:
        struct Result {
            Float sqrDistance;
            Float lineParameter;
        };

        // Return the sqrDistance
        Float operator()(Ray const &line, BoundingBox const &box) {
            // Translate the line and box so that the box has center at the origin.
            Vector boxCenter = Vector(box.center());
            Vector boxExtent = box.extents();
            Vector point = Vector(line.o - boxCenter);
            Vector direction = line.d;

            Result result;
            DoQuery(point, direction, boxExtent, result);

            if(result.lineParameter < 0) {
                return box.squared_distance(line.o);
            } else if(result.lineParameter > line.maxt) {
                const Point p = line(line.maxt-line.mint);
                return box.squared_distance(p);
            } else {
                return result.sqrDistance;
            }
        }

    protected:
        // Compute the distance and closest point between a line and an
        // axis-aligned box whose center is the origin.  On input, 'point' is the
        // line origin and 'direction' is the line direction.  On output, 'point'
        // is the point on the box closest to the line.  The 'direction' is
        // non-const to allow transforming the problem into the first octant.
        void DoQuery(Vector &point, Vector &direction,
                     Vector const &boxExtent, Result &result) {
            result.sqrDistance = (Float) 0;
            result.lineParameter = (Float) 0;

            // Apply reflections so that direction vector has nonnegative components.
            bool reflect[3];
            for (
                    int i = 0;
                    i < 3; ++i) {
                if (direction[i] < (Float) 0) {
                    point[i] = -point[i];
                    direction[i] = -direction[i];
                    reflect[i] = true;
                } else {
                    reflect[i] = false;
                }
            }

            if (direction[0] > (Float) 0) {
                if (direction[1] > (Float) 0) {
                    if (direction[2] > (Float) 0)  // (+,+,+)
                    {
                        CaseNoZeros(point, direction, boxExtent, result);
                    } else  // (+,+,0)
                    {
                        Case0(0, 1, 2, point, direction, boxExtent, result);
                    }
                } else {
                    if (direction[2] > (Float) 0)  // (+,0,+)
                    {
                        Case0(0, 2, 1, point, direction, boxExtent, result);
                    } else  // (+,0,0)
                    {
                        Case00(0, 1, 2, point, direction, boxExtent, result);
                    }
                }
            } else {
                if (direction[1] > (Float) 0) {
                    if (direction[2] > (Float) 0)  // (0,+,+)
                    {
                        Case0(1, 2, 0, point, direction, boxExtent, result);
                    } else  // (0,+,0)
                    {
                        Case00(1, 0, 2, point, direction, boxExtent, result);
                    }
                } else {
                    if (direction[2] > (Float) 0)  // (0,0,+)
                    {
                        Case00(2, 0, 1, point, direction, boxExtent, result);
                    } else  // (0,0,0)
                    {
                        Case000(point, boxExtent, result);
                    }
                }
            }

            // Undo the reflections applied previously.
            for (int i = 0; i < 3; ++i) {
                if (reflect[i]) {
                    point[i] = -point[i];
                }
            }
        }

    private:

        void Face(int i0, int i1, int i2, Vector &pnt,
                  Vector const &dir, Vector const &PmE,
                  Vector const &boxExtent, Result &result) {
            Vector PpE;
            Float lenSqr, inv, tmp, param, t, delta;

            PpE[i1] = pnt[i1] + boxExtent[i1];
            PpE[i2] = pnt[i2] + boxExtent[i2];
            if (dir[i0] * PpE[i1] >= dir[i1] * PmE[i0]) {
                if (dir[i0] * PpE[i2] >= dir[i2] * PmE[i0]) {
                    // v[i1] >= -e[i1], v[i2] >= -e[i2] (distance = 0)
                    pnt[i0] = boxExtent[i0];
                    inv = ((Float) 1) / dir[i0];
                    pnt[i1] -= dir[i1] * PmE[i0] *
                               inv;
                    pnt[i2] -= dir[i2] * PmE[i0] *
                               inv;
                    result.lineParameter = -PmE[i0] * inv;
                } else {
                    // v[i1] >= -e[i1], v[i2] < -e[i2]
                    lenSqr = dir[i0] * dir[i0] + dir[i2] * dir[i2];
                    tmp = lenSqr * PpE[i1] - dir[i1] * (dir[i0] * PmE[i0] +
                                                        dir[i2] * PpE[i2]);
                    if (tmp <= ((Float) 2) *
                               lenSqr * boxExtent[i1]
                            ) {
                        t = tmp / lenSqr;
                        lenSqr += dir[i1] * dir[i1];
                        tmp = PpE[i1] - t;
                        delta = dir[i0] * PmE[i0] + dir[i1] * tmp + dir[i2] * PpE[i2];
                        param = -delta / lenSqr;
                        result.sqrDistance += PmE[i0] * PmE[i0] +
                                              tmp * tmp
                                              +
                                              PpE[i2] * PpE[i2] +
                                              delta * param;

                        result.lineParameter = param;
                        pnt[i0] = boxExtent[i0];
                        pnt[i1] = t - boxExtent[i1];
                        pnt[i2] = -boxExtent[i2];
                    } else {
                        lenSqr += dir[i1] * dir[i1];
                        delta = dir[i0] * PmE[i0] + dir[i1] * PmE[i1] + dir[i2] * PpE[i2];
                        param = -delta / lenSqr;
                        result.sqrDistance += PmE[i0] * PmE[i0] + PmE[i1] * PmE[i1]
                                              + PpE[i2] * PpE[i2] +
                                              delta * param;

                        result.lineParameter = param;
                        pnt[i0] = boxExtent[i0];
                        pnt[i1] = boxExtent[i1];
                        pnt[i2] = -boxExtent[i2];
                    }
                }
            } else {
                if (dir[i0] * PpE[i2] >= dir[i2] * PmE[i0]) {
                    // v[i1] < -e[i1], v[i2] >= -e[i2]
                    lenSqr = dir[i0] * dir[i0] + dir[i1] * dir[i1];
                    tmp = lenSqr * PpE[i2] - dir[i2] * (dir[i0] * PmE[i0] +
                                                        dir[i1] * PpE[i1]);
                    if (tmp <= ((Float) 2) *
                               lenSqr * boxExtent[i2]
                            ) {
                        t = tmp / lenSqr;
                        lenSqr += dir[i2] * dir[i2];
                        tmp = PpE[i2] - t;
                        delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * tmp;
                        param = -delta / lenSqr;
                        result.sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1] +
                                              tmp * tmp
                                              +
                                              delta * param;

                        result.lineParameter = param;
                        pnt[i0] = boxExtent[i0];
                        pnt[i1] = -boxExtent[i1];
                        pnt[i2] = t - boxExtent[i2];
                    } else {
                        lenSqr += dir[i2] * dir[i2];
                        delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * PmE[i2];
                        param = -delta / lenSqr;
                        result.sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1] +
                                              PmE[i2] * PmE[i2] +
                                              delta * param;

                        result.
                                lineParameter = param;
                        pnt[i0] = boxExtent[i0];
                        pnt[i1] = -boxExtent[i1];
                        pnt[i2] = boxExtent[i2];
                    }
                } else {
                    // v[i1] < -e[i1], v[i2] < -e[i2]
                    lenSqr = dir[i0] * dir[i0] + dir[i2] * dir[i2];
                    tmp = lenSqr * PpE[i1] - dir[i1] * (dir[i0] * PmE[i0] +
                                                        dir[i2] * PpE[i2]);
                    if (tmp >= (Float) 0) {
                        // v[i1]-edge is closest
                        if (tmp <= ((Float) 2) *
                                   lenSqr * boxExtent[i1]
                                ) {
                            t = tmp / lenSqr;
                            lenSqr += dir[i1] * dir[i1];
                            tmp = PpE[i1] - t;
                            delta = dir[i0] * PmE[i0] + dir[i1] * tmp + dir[i2] * PpE[i2];
                            param = -delta / lenSqr;
                            result.sqrDistance += PmE[i0] * PmE[i0] +
                                                  tmp * tmp
                                                  +
                                                  PpE[i2] * PpE[i2] +
                                                  delta * param;

                            result.lineParameter = param;
                            pnt[i0] = boxExtent[i0];
                            pnt[i1] = t - boxExtent[i1];
                            pnt[i2] = -boxExtent[i2];
                        } else {
                            lenSqr += dir[i1] * dir[i1];
                            delta = dir[i0] * PmE[i0] + dir[i1] * PmE[i1]
                                    + dir[i2] * PpE[i2];
                            param = -delta / lenSqr;
                            result.sqrDistance += PmE[i0] * PmE[i0] + PmE[i1] * PmE[i1]
                                                  + PpE[i2] * PpE[i2] +
                                                  delta * param;

                            result.lineParameter = param;
                            pnt[i0] = boxExtent[i0];
                            pnt[i1] = boxExtent[i1];
                            pnt[i2] = -boxExtent[i2];
                        }
                        return;
                    }

                    lenSqr = dir[i0] * dir[i0] + dir[i1] * dir[i1];
                    tmp = lenSqr * PpE[i2] - dir[i2] * (dir[i0] * PmE[i0] +
                                                        dir[i1] * PpE[i1]);
                    if (tmp >= (Float) 0) {
                        // v[i2]-edge is closest
                        if (tmp <= ((Float) 2) *
                                   lenSqr * boxExtent[i2]
                                ) {
                            t = tmp / lenSqr;
                            lenSqr += dir[i2] * dir[i2];
                            tmp = PpE[i2] - t;
                            delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * tmp;
                            param = -delta / lenSqr;
                            result.sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1] +
                                                  tmp * tmp
                                                  +
                                                  delta * param;

                            result.lineParameter = param;
                            pnt[i0] = boxExtent[i0];
                            pnt[i1] = -boxExtent[i1];
                            pnt[i2] = t - boxExtent[i2];
                        } else {
                            lenSqr += dir[i2] * dir[i2];
                            delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1]
                                    + dir[i2] * PmE[i2];
                            param = -delta / lenSqr;
                            result.sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1]
                                                  + PmE[i2] * PmE[i2] +
                                                  delta * param;

                            result.lineParameter = param;
                            pnt[i0] = boxExtent[i0];
                            pnt[i1] = -boxExtent[i1];
                            pnt[i2] = boxExtent[i2];
                        }
                        return;
                    }

                    // (v[i1],v[i2])-corner is closest
                    lenSqr += dir[i2] * dir[i2];
                    delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * PpE[i2];
                    param = -delta / lenSqr;
                    result.sqrDistance += PmE[i0] * PmE[i0] + PpE[i1] * PpE[i1]
                                          + PpE[i2] * PpE[i2] +
                                          delta * param;

                    result.lineParameter = param;
                    pnt[i0] = boxExtent[i0];
                    pnt[i1] = -boxExtent[i1];
                    pnt[i2] = -boxExtent[i2];
                }
            }
        }

        void CaseNoZeros(Vector &pnt, Vector const &dir,
                         Vector const &boxExtent, Result &result) {
            Vector PmE = pnt - boxExtent;
            Float prodDxPy = dir[0] * PmE[1];
            Float prodDyPx = dir[1] * PmE[0];
            Float prodDzPx, prodDxPz, prodDzPy, prodDyPz;

            if (prodDyPx >= prodDxPy) {
                prodDzPx = dir[2] * PmE[0];
                prodDxPz = dir[0] * PmE[2];
                if (prodDzPx >= prodDxPz) {
                    // line intersects x = e0
                    Face(0, 1, 2, pnt, dir, PmE, boxExtent, result);
                } else {
                    // line intersects z = e2
                    Face(2, 0, 1, pnt, dir, PmE, boxExtent, result);
                }
            } else {
                prodDzPy = dir[2] * PmE[1];
                prodDyPz = dir[1] * PmE[2];
                if (prodDzPy >= prodDyPz) {
                    // line intersects y = e1
                    Face(1, 2, 0, pnt, dir, PmE, boxExtent, result);
                } else {
                    // line intersects z = e2
                    Face(2, 0, 1, pnt, dir, PmE, boxExtent, result);
                }
            }
        }

        void Case0(int i0, int i1, int i2, Vector &pnt,
                   Vector const &dir, Vector const &boxExtent,
                   Result &result) {
            Float PmE0 = pnt[i0] - boxExtent[i0];
            Float PmE1 = pnt[i1] - boxExtent[i1];
            Float prod0 = dir[i1] * PmE0;
            Float prod1 = dir[i0] * PmE1;
            Float delta, invLSqr, inv;

            if (prod0 >= prod1) {
                // line intersects P[i0] = e[i0]
                pnt[i0] = boxExtent[i0];

                Float PpE1 = pnt[i1] + boxExtent[i1];
                delta = prod0 - dir[i0] * PpE1;
                if (delta >= (Float) 0) {
                    invLSqr = ((Float) 1) / (dir[i0] * dir[i0] + dir[i1] * dir[i1]);
                    result.sqrDistance +=
                            delta * delta
                            *
                            invLSqr;
                    pnt[i1] = -boxExtent[i1];
                    result.
                            lineParameter = -(dir[i0] * PmE0 + dir[i1] * PpE1) * invLSqr;
                } else {
                    inv = ((Float) 1) / dir[i0];
                    pnt[i1] -=
                            prod0 * inv;
                    result.
                            lineParameter = -PmE0 * inv;
                }
            } else {
                // line intersects P[i1] = e[i1]
                pnt[i1] = boxExtent[i1];

                Float PpE0 = pnt[i0] + boxExtent[i0];
                delta = prod1 - dir[i1] * PpE0;
                if (delta >= (Float) 0) {
                    invLSqr = ((Float) 1) / (dir[i0] * dir[i0] + dir[i1] * dir[i1]);
                    result.sqrDistance +=
                            delta * delta
                            *
                            invLSqr;
                    pnt[i0] = -boxExtent[i0];
                    result.
                            lineParameter = -(dir[i0] * PpE0 + dir[i1] * PmE1) * invLSqr;
                } else {
                    inv = ((Float) 1) / dir[i1];
                    pnt[i0] -=
                            prod1 * inv;
                    result.
                            lineParameter = -PmE1 * inv;
                }
            }

            if (pnt[i2] < -boxExtent[i2]) {
                delta = pnt[i2] + boxExtent[i2];
                result.sqrDistance +=
                        delta * delta;
                pnt[i2] = -boxExtent[i2];
            } else if (pnt[i2] > boxExtent[i2]) {
                delta = pnt[i2] - boxExtent[i2];
                result.sqrDistance +=
                        delta * delta;
                pnt[i2] = boxExtent[i2];
            }
        }

        void Case00(int i0, int i1, int i2, Vector &pnt,
                    Vector const &dir, Vector const &boxExtent,
                    Result &result) {
            Float delta;

            result.
                    lineParameter = (boxExtent[i0] - pnt[i0]) / dir[i0];

            pnt[i0] = boxExtent[i0];

            if (pnt[i1] < -boxExtent[i1]) {
                delta = pnt[i1] + boxExtent[i1];
                result.sqrDistance +=
                        delta * delta;
                pnt[i1] = -boxExtent[i1];
            } else if (pnt[i1] > boxExtent[i1]) {
                delta = pnt[i1] - boxExtent[i1];
                result.sqrDistance +=
                        delta * delta;
                pnt[i1] = boxExtent[i1];
            }

            if (pnt[i2] < -boxExtent[i2]) {
                delta = pnt[i2] + boxExtent[i2];
                result.sqrDistance +=
                        delta * delta;
                pnt[i2] = -boxExtent[i2];
            } else if (pnt[i2] > boxExtent[i2]) {
                delta = pnt[i2] - boxExtent[i2];
                result.sqrDistance +=
                        delta * delta;
                pnt[i2] = boxExtent[i2];
            }
        }

        void Case000(Vector &pnt, Vector const &boxExtent,
                     Result &result) {
            Float delta;

            if (pnt[0] < -boxExtent[0]) {
                delta = pnt[0] + boxExtent[0];
                result.sqrDistance +=
                        delta * delta;
                pnt[0] = -boxExtent[0];
            } else if (pnt[0] > boxExtent[0]) {
                delta = pnt[0] - boxExtent[0];
                result.sqrDistance +=
                        delta * delta;
                pnt[0] = boxExtent[0];
            }

            if (pnt[1] < -boxExtent[1]) {
                delta = pnt[1] + boxExtent[1];
                result.sqrDistance +=
                        delta * delta;
                pnt[1] = -boxExtent[1];
            } else if (pnt[1] > boxExtent[1]) {
                delta = pnt[1] - boxExtent[1];
                result.sqrDistance +=
                        delta * delta;
                pnt[1] = boxExtent[1];
            }

            if (pnt[2] < -boxExtent[2]) {
                delta = pnt[2] + boxExtent[2];
                result.sqrDistance +=
                        delta * delta;
                pnt[2] = -boxExtent[2];
            } else if (pnt[2] > boxExtent[2]) {
                delta = pnt[2] - boxExtent[2];
                result.sqrDistance +=
                        delta * delta;
                pnt[2] = boxExtent[2];
            }
        }
    };

    auto [valid, _v1, _v2] = ray_intersect(ray);
    if(valid){
        return 0.0f;
    } else {
        // This code does not handle the min distance
        MinDistanceAABB query;
        return query.operator()(ray, *this);
    }

}


    Point min; ///< Component-wise minimum
    Point max; ///< Component-wise maximum
};

/// Print a string representation of the bounding box
template <typename Point>
std::ostream &operator<<(std::ostream &os, const BoundingBox<Point> &bbox) {
    os << "BoundingBox" << type_suffix<Point>();
    if (all(!bbox.valid()))
        os << "[invalid]";
    else
        os << "[" << std::endl
           << "  min = " << bbox.min << "," << std::endl
           << "  max = " << bbox.max << std::endl
           << "]";
    return os;
}

NAMESPACE_END(mitsuba)
