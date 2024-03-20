const std = @import("std");
const tracy = @import("tracy");
const expectEqual = std.testing.expectEqual;

const Allocator = std.mem.Allocator;
pub const Precision = f32;

pub const VectorSize = std.simd.suggestVectorLength(Precision).?;
pub fn vectorType(comptime len: comptime_int) type {
    return @Vector(len, Precision);
}
pub const FullVector = @Vector(VectorSize, Precision);
pub const BoolVector = @Vector(VectorSize, bool);

fn one_vector_sigmoid(comptime len: comptime_int, vector: vectorType(len)) vectorType(len) {
    const ones: vectorType(len) = @splat(1.0);
    return ones / (ones + @exp(-vector));
}

/// Returns the scalar result of multiplying piecewise components of A and B, and summing their results.
fn scalar_product(comptime len: comptime_int, A: *const [len]Precision, B: *const [len]Precision) Precision {
    const zone = tracy.initZone(@src(), .{ .name = "Scalar Product" });
    defer zone.deinit();
    const tail_len: usize = len % VectorSize;
    const body_len: usize = len - tail_len;
    // std.debug.print("Begin\n\n", .{});
    var i: usize = 0;
    var res: Precision = 0.0;
    while (i < body_len) : (i += VectorSize) {
        const vec1: FullVector = A[i..][0..VectorSize].*;
        const vec2: FullVector = B[i..][0..VectorSize].*;
        // std.debug.print("vec1 * vec2 = {d} * {d} = {d}, then add them and get {d}\n", .{ vec1, vec2, vec1 * vec2, @reduce(.Add, vec1 * vec2) });
        res += @reduce(.Add, vec1 * vec2);
    }
    for (body_len..len) |j| {
        res += A[j] * B[j];
    }
    // std.debug.print("End.  Got {d}\n\n", .{res});
    return res;
}

pub fn model(comptime num_features: comptime_int, comptime num_samples: comptime_int) type {
    const Sample = [num_features]Precision;
    const Parameters = [num_features]Precision;
    const features_tail = num_features % VectorSize;
    const features_body = num_features - features_tail;
    const samples_tail = num_samples % VectorSize;
    const samples_body = num_samples - samples_tail;

    const Prediction = struct {
        predictions: *[num_samples]Precision,
        const Self = @This();
        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.destroy(self.predictions);
        }
    };
    const PropagateResult = struct {
        dw: *Parameters,
        db: Precision,
        cost: Precision,
        const Self = @This();
        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.destroy(self.dw);
        }
    };
    const OptimizeResult = struct {
        w: *Parameters,
        // dw: *Parameters,
        // db: Precision,
        cost: Precision,
        b: Precision,
        const Self = @This();
        pub fn deinit(self: Self, allocator: Allocator) void {
            // allocator.destroy(self.dw);
            allocator.destroy(self.w);
        }
    };
    const TestResult = struct { accuracy: Precision, results: [num_samples]bool };
    const SamplesTailVector = vectorType(samples_tail);
    const FeaturesTailVector = vectorType(features_tail);
    const Ones: FullVector = @splat(1.0);
    const samples_tail_ones: SamplesTailVector = @splat(1.0);

    return struct {
        X: *const [num_samples]Sample,
        Y: *const [num_samples]bool,
        const Self = @This();

        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.free(self.X);
            allocator.free(self.Y);
        }
        /// Returns the number of values in self.Y that are correctly
        pub fn testParameters(self: Self, allocator: Allocator, parameters: *Parameters, bias: Precision) !TestResult {
            var res: [num_samples]bool = undefined;
            const prediction = try self.getPrediction(allocator, parameters, bias);
            defer prediction.deinit(allocator);
            var total: Precision = 0.0;
            for (prediction.predictions, 0..) |value, i| {
                const guess = value >= 0.5;
                res[i] = guess == self.Y[i];
                if (res[i]) {
                    total += 1.0;
                }
            }
            return TestResult{ .accuracy = total / num_samples, .results = res };
        }
        /// Predicts the outcomes for each sample in X, given the parameters (w) and bias (b).
        /// This is defined as $\hat{y} = \sigma(w^{T}X + b)$.
        /// We'll do it once, quick and dirty, and see if we can see a route to using less memory
        /// in the computation.
        pub fn getPrediction(self: Self, allocator: Allocator, parameters: *Parameters, bias: Precision) !Prediction {
            const zone = tracy.initZone(@src(), .{ .name = "Get Prediction" });
            defer zone.deinit();

            var result = try allocator.create([num_samples]Precision);
            errdefer allocator.destroy(result);
            for (0..num_samples) |i| {
                result[i] = scalar_product(num_features, parameters, &self.X[i]);
            }

            var i: usize = 0;
            const b: FullVector = @splat(bias);
            while (i < samples_body) : (i += VectorSize) {
                const vec1: FullVector = result[i..][0..VectorSize].*;
                const vec2: FullVector = Ones / (Ones + @exp(-vec1 - b));
                result[i..][0..VectorSize].* = vec2;
            }
            const b1: SamplesTailVector = @splat(bias);
            const vec1: SamplesTailVector = result[samples_body..num_samples].*;
            const vec2: SamplesTailVector = samples_tail_ones / (samples_tail_ones + @exp(-vec1 - b1));
            result[samples_body..num_samples].* = vec2;

            return Prediction{ .predictions = result };
        }
        pub fn propagate(self: Self, allocator: Allocator, parameters: *Parameters, bias: Precision) !PropagateResult {
            const zone = tracy.initZone(@src(), .{ .name = "Propagate Fn" });
            defer zone.deinit();
            const prediction: Prediction = try self.getPrediction(allocator, parameters, bias);
            defer prediction.deinit(allocator);
            var a_minus_y: *[num_samples]Precision = try allocator.create([num_samples]Precision);
            var cost: Precision = 0.0;
            var db: Precision = 0.0;
            defer allocator.destroy(a_minus_y);
            {
                for (self.Y, 0..) |is_class, idx| {
                    a_minus_y[idx] = prediction.predictions[idx] - @as(Precision, if (is_class) 1 else 0);
                    db += a_minus_y[idx];
                    cost -= switch (is_class) {
                        true => @log(prediction.predictions[idx]),
                        false => @log(1.0 - prediction.predictions[idx]),
                    };
                }
                db /= num_samples;
                cost /= num_samples;
            }

            var dw: *Parameters = try allocator.create(Parameters);
            errdefer allocator.destroy(dw);
            {
                const calc_dw_zone = tracy.initZone(@src(), .{ .name = "Backward Propagation" });
                defer calc_dw_zone.deinit();
                for (0..num_features) |feat_idx| {
                    // Do a scalar product in-place to avoid copying data around too much.
                    // const dw_feature_zone = tracy.initZone(@src(), .{ .name = "Back propagating a single feature" });
                    // defer dw_feature_zone.deinit();
                    var sample_idx: usize = 0;
                    var res: Precision = 0.0;
                    while (sample_idx < samples_body) : (sample_idx += VectorSize) {
                        const vec_zone = tracy.initZone(@src(), .{ .name = "Build vec1" });
                        var vec1: FullVector = undefined;
                        inline for (0..VectorSize) |i| {
                            vec1[i] = self.X[sample_idx + i][feat_idx];
                        }
                        // const vec2_zone = tracy.initZone(@src(), .{ .name = "Get second vector and reduce" });
                        // defer vec2_zone.deinit();
                        const vec2: FullVector = a_minus_y[sample_idx..][0..VectorSize].*;
                        vec_zone.deinit();
                        res += @reduce(.Add, vec1 * vec2);
                    }
                    dw[feat_idx] = res / num_samples;
                }
            }
            return PropagateResult{ .dw = dw, .cost = cost, .db = db };
        }
        pub fn optimize(self: Self, allocator: Allocator, num_iterations: u32, learning_rate: Precision, comptime print_res: bool) !OptimizeResult {
            const optimize_zone = tracy.initZone(@src(), .{ .name = "Inside optimize" });
            defer optimize_zone.deinit();

            var w: *Parameters = try allocator.create(Parameters);
            errdefer allocator.destroy(w);
            for (0..num_features) |i| {
                w[i] = 0.0;
            }
            // const stdout = std.io.getStdOut().writer();

            const alpha: FullVector = @splat(learning_rate);
            var b: Precision = 0.0;
            var cost: Precision = 0.0;
            for (0..num_iterations) |iter| {
                const loop_zone = tracy.initZone(@src(), .{ .name = "Optimize Loop" });
                defer loop_zone.deinit();

                const prop = try self.propagate(allocator, w, b);
                defer prop.deinit(allocator);
                cost = prop.cost;
                b -= learning_rate * prop.db;
                const calc_w_zone = tracy.initZone(@src(), .{ .name = "Calculate dw" });
                var i: usize = 0;
                while (i < features_body) : (i += VectorSize) {
                    const wi = w[i..];
                    const dwi = prop.dw[i..];
                    const vec1: FullVector = wi[0..VectorSize].*;
                    const vec2: FullVector = dwi[0..VectorSize].*;
                    wi[0..VectorSize].* = vec1 - (alpha * vec2);
                }
                const vec1: FeaturesTailVector = w[features_body..num_features].*;
                const vec2: FeaturesTailVector = prop.dw[features_body..num_features].*;
                w[features_body..num_features].* = vec1 - (vec2 * @as(FeaturesTailVector, @splat(learning_rate)));
                calc_w_zone.deinit();
                if (print_res and iter % 100 == 0) {
                    std.debug.print("Finished {d} iterations.  Cost is {d}\n", .{ iter, cost });
                }
            }
            return OptimizeResult{ .cost = cost, .b = b, .w = w };
        }
        fn sigmoid(input: Sample, output: *Sample) void {
            // @breakpoint();
            var i: usize = 0;
            // @breakpoint();
            while (i < features_body) : (i += VectorSize) {
                const vec = read_vec(VectorSize, input[i .. i + VectorSize]);
                // @breakpoint();
                const vec2 = one_vector_sigmoid(VectorSize, vec);
                // @breakpoint();
                write_vec(VectorSize, vec2, output[i .. i + VectorSize]);
            }
            const vec = read_vec(features_tail, input[features_body..num_features]);
            const vec2 = one_vector_sigmoid(features_tail, vec);
            write_vec(features_tail, vec2, output[features_body..num_features]);
        }
    };
}

pub fn main() !void {}

test "Basic test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const Struct = model(2, 3);
    const my_model = Struct{ .X = try allocator.create([3][2]Precision), .Y = try allocator.create([3]bool) };
    defer my_model.deinit(allocator);
}
test {
    std.testing.refAllDecls(@This());
    _ = TestScalarProduct;
}
const TestScalarProduct = struct {
    var rng = std.rand.DefaultPrng.init(123456324);
    fn make_float_array(comptime size: u8) [size]Precision {
        var res: [size]Precision = undefined;
        for (0..size) |i| {
            res[i] = @floatCast(rng.random().float(f32));
        }
        return res;
    }
    test "Scalar Product of two arrays" {
        const size = 15;
        const A: [size]Precision = .{ 0.7935, 0.1582, 0.876, 0.615, 0.8623, 0.283, 0.7397, 0.1952, 0.1895, 0.4993, 0.5454, 0.0001229, 0.3447, 0.381, 0.845 };
        const B: [size]Precision = .{ 0.4976, 0.1895, 0.804, 0.6753, 0.9673, 0.631, 0.43, 0.272, 0.971, 0.02385, 0.138, 0.74, 0.53, 0.1631, 0.7646 };
        // const C: [2][size]Precision = .{ .{ 0.7935, 0.1582, 0.876, 0.615, 0.8623, 0.283, 0.7397, 0.1952, 0.1895, 0.4993, 0.5454, 0.0001229, 0.3447, 0.381, 0.845 }, .{ 0.7935, 0.1582, 0.876, 0.615, 0.8623, 0.283, 0.7397, 0.1952, 0.1895, 0.4993, 0.5454, 0.0001229, 0.3447, 0.381, 0.845 } };

        // std.debug.print("A = {d}\nB = {d}\nC = {d}\n", .{ A, B, C });
        const c = scalar_product(size, &A, &B);
        try expectEqual(4.094, c);
        // 4.090467241
    }
    const sample_data: [29][37]Precision = .{
        .{ 0.7935, 0.1582, 0.876, 0.615, 0.8623, 0.283, 0.7397, 0.1952, 0.1895, 0.4993, 0.5454, 0.0001229, 0.3447, 0.381, 0.845, 0.4976, 0.1895, 0.804, 0.6753, 0.9673, 0.631, 0.43, 0.272, 0.971, 0.02385, 0.138, 0.74, 0.53, 0.1631, 0.7646, 0.61, 0.802, 0.8765, 0.459, 0.776, 0.9487, 0.1037 },
        .{ 0.3108, 0.5454, 0.577, 0.2041, 0.769, 0.3784, 0.0423, 0.2131, 0.387, 0.4138, 0.0725, 0.393, 0.439, 0.8296, 0.9795, 0.426, 0.2844, 0.2983, 0.9146, 0.315, 0.6865, 0.5356, 0.8853, 0.1965, 0.973, 0.1938, 0.8994, 0.349, 0.638, 0.5312, 0.128, 0.7656, 0.5186, 0.9365, 0.3386, 0.0851, 0.1812 },
        .{ 0.779, 0.5503, 0.7925, 0.003242, 0.88, 0.727, 0.3987, 0.482, 0.635, 0.422, 0.5215, 0.5205, 0.3618, 0.6074, 0.9067, 0.4983, 0.2668, 0.95, 0.509, 0.3323, 0.3567, 0.2196, 0.9595, 0.01764, 0.6177, 0.1037, 0.06082, 0.4731, 0.283, 0.2277, 0.4033, 0.4, 0.7065, 0.4023, 0.4468, 0.568, 0.6143 },
        .{ 0.04416, 0.333, 0.07776, 0.576, 0.7676, 0.2128, 0.5415, 0.729, 0.8975, 0.1455, 0.9897, 0.8965, 0.5405, 0.345, 0.8755, 0.331, 0.781, 0.9033, 0.02362, 0.3442, 0.836, 0.192, 0.7207, 0.02287, 0.6167, 0.6377, 0.3545, 0.1743, 0.02173, 0.0402, 0.318, 0.54, 0.793, 0.1852, 0.913, 0.6973, 0.1464 },
        .{ 0.5947, 0.3726, 0.1552, 0.6885, 0.7847, 0.9414, 0.1969, 0.2446, 0.163, 0.10614, 0.08466, 0.882, 0.8394, 0.2878, 0.8193, 0.969, 0.4426, 0.6045, 0.1903, 0.02135, 0.01718, 0.1843, 0.606, 0.617, 0.6997, 0.3413, 0.447, 0.773, 0.7666, 0.01543, 0.4507, 0.2974, 0.8716, 0.608, 0.736, 0.617, 0.5796 },
        .{ 0.8438, 0.9927, 0.8247, 0.1512, 0.784, 0.14, 0.26, 0.987, 0.2295, 0.9033, 0.794, 0.862, 0.925, 0.1711, 0.7583, 0.6826, 0.2124, 0.2261, 0.8467, 0.1422, 0.9805, 0.693, 0.2798, 0.622, 0.2817, 0.96, 0.07495, 0.5693, 0.507, 0.9463, 0.405, 0.1393, 0.5547, 0.8726, 0.5894, 0.583, 0.08716 },
        .{ 0.6514, 0.871, 0.02623, 0.2507, 0.88, 0.9585, 0.718, 0.3816, 0.903, 0.939, 0.933, 0.6675, 0.3774, 0.3586, 0.5127, 0.6035, 0.8735, 0.6113, 0.309, 0.2515, 0.1211, 0.1223, 0.4888, 0.5884, 0.8774, 0.156, 0.8496, 0.8965, 0.6353, 0.4158, 0.828, 0.815, 0.829, 0.4124, 0.8584, 0.26, 0.8765 },
        .{ 0.5293, 0.1626, 0.4797, 0.5054, 0.647, 0.524, 0.387, 0.1384, 0.621, 0.793, 0.2151, 0.7524, 0.1749, 0.5825, 0.6035, 0.7485, 0.1071, 0.5845, 0.08167, 0.4294, 0.518, 0.1597, 0.131, 0.981, 0.848, 0.0392, 0.912, 0.371, 0.3547, 0.3398, 0.9424, 0.817, 0.03754, 0.0659, 0.3357, 0.917, 0.8647 },
        .{ 0.9575, 0.1311, 0.9536, 0.7544, 0.4768, 0.5146, 0.5117, 0.1163, 0.8384, 0.5083, 0.1056, 0.3909, 0.1769, 0.872, 0.718, 0.7227, 0.507, 0.1757, 0.3044, 0.1758, 0.4275, 0.3328, 0.3542, 0.3079, 0.977, 0.8853, 0.1032, 0.9883, 0.569, 0.702, 0.587, 0.0782, 0.9307, 0.965, 0.792, 0.9824, 0.694 },
        .{ 0.6455, 0.209, 0.1644, 0.09436, 0.525, 0.902, 0.5127, 0.2695, 0.384, 0.1881, 0.822, 0.474, 0.8154, 0.625, 0.5117, 0.9165, 0.2334, 0.977, 0.8354, 0.00346, 0.3848, 0.064, 0.0445, 0.3, 0.0993, 0.5254, 0.3442, 0.613, 0.487, 0.5703, 0.843, 0.1794, 0.01374, 0.618, 0.2507, 0.519, 0.0496 },
        .{ 0.9795, 0.5625, 0.892, 0.7993, 0.3682, 0.3462, 0.807, 0.8047, 0.8867, 0.1367, 0.4084, 0.85, 0.03848, 0.02357, 0.09247, 0.3677, 0.8315, 0.4868, 0.2091, 0.1798, 0.4397, 0.02928, 0.9985, 0.2491, 0.7705, 0.2954, 0.9204, 0.302, 0.923, 0.6675, 0.958, 0.1333, 0.5093, 0.543, 0.0693, 0.7563, 0.925 },
        .{ 0.539, 0.502, 0.564, 0.677, 0.3203, 0.75, 0.5444, 0.9604, 0.591, 0.7896, 0.0272, 0.0777, 0.612, 0.2365, 0.4392, 0.3127, 0.3875, 0.726, 0.3484, 0.9004, 0.3064, 0.709, 0.538, 0.457, 0.427, 0.1967, 0.982, 0.3044, 0.5605, 0.74, 0.958, 0.527, 0.7246, 0.898, 0.05585, 0.7617, 0.7925 },
        .{ 0.0937, 0.0488, 0.8076, 0.3445, 0.3772, 0.2231, 0.773, 0.3325, 0.8223, 0.581, 0.1509, 0.986, 0.4001, 0.4277, 0.879, 0.1902, 0.003714, 0.05286, 0.419, 0.7656, 0.6724, 0.0588, 0.9565, 0.808, 0.447, 0.568, 0.5986, 0.7485, 0.02005, 0.2314, 0.1747, 0.55, 0.3525, 0.12396, 0.2793, 0.651, 0.5513 },
        .{ 0.2947, 0.784, 0.1375, 0.4321, 0.7134, 0.3247, 0.577, 0.7026, 0.6943, 0.0659, 0.409, 0.957, 0.004673, 0.936, 0.775, 0.1145, 0.2133, 0.408, 0.1313, 0.3389, 0.09247, 0.7485, 0.4944, 0.3271, 0.6836, 0.8447, 0.04767, 0.279, 0.5586, 0.824, 0.65, 0.2325, 0.4524, 0.1412, 0.3538, 0.2922, 0.326 },
        .{ 0.995, 0.6543, 0.8296, 0.0624, 0.2961, 0.8926, 0.4055, 0.848, 0.281, 0.2018, 0.1025, 0.02196, 0.187, 0.565, 0.632, 0.929, 0.3545, 0.846, 0.01527, 0.1092, 0.992, 0.2308, 0.2416, 0.1292, 0.775, 0.7656, 0.52, 0.844, 0.433, 0.7773, 0.4155, 0.743, 0.2837, 0.957, 0.9067, 0.966, 0.0325 },
        .{ 0.874, 0.3137, 0.633, 0.851, 0.9937, 0.392, 0.1575, 0.801, 0.2325, 0.4858, 0.6245, 0.08636, 0.733, 0.5625, 0.1774, 0.1222, 0.88, 0.655, 0.8193, 0.308, 0.3303, 0.3176, 0.8486, 0.552, 0.5503, 0.4316, 0.2089, 0.996, 0.771, 0.2866, 0.5806, 0.0892, 0.3877, 0.06683, 0.9766, 0.507, 0.482 },
        .{ 0.0667, 0.4978, 0.8755, 0.8086, 0.965, 0.2338, 0.227, 0.6807, 0.677, 0.4631, 0.03238, 0.10004, 0.6973, 0.654, 0.2203, 0.6133, 0.449, 0.8315, 0.8296, 0.4685, 0.4417, 0.814, 0.2961, 0.4268, 0.00747, 0.3708, 0.7563, 0.3562, 0.695, 0.8735, 0.1035, 0.684, 0.953, 0.1285, 0.3613, 0.634, 0.9 },
        .{ 0.7764, 0.7705, 0.7705, 0.42, 0.02753, 0.984, 0.6934, 0.2615, 0.235, 0.4365, 0.796, 0.76, 0.872, 0.6533, 0.872, 0.959, 0.3513, 0.9316, 0.8003, 0.873, 0.5815, 0.1422, 0.435, 0.8228, 0.03143, 0.9336, 0.4375, 0.3335, 0.3413, 0.1946, 0.9375, 0.3013, 0.668, 0.1245, 0.1298, 0.9834, 0.6123 },
        .{ 0.5557, 0.3315, 0.1505, 0.2074, 0.3455, 0.1608, 0.523, 0.9688, 0.983, 0.735, 0.1095, 0.3506, 0.1323, 0.7104, 0.825, 0.593, 0.7744, 0.89, 0.5312, 0.1449, 0.04846, 0.374, 0.3213, 0.9395, 0.6494, 0.649, 0.1271, 0.7393, 0.3843, 0.09503, 0.5923, 0.9585, 0.739, 0.2913, 0.946, 0.93, 0.4128 },
        .{ 0.579, 0.7383, 0.0342, 0.891, 0.1236, 0.9097, 0.6978, 0.7715, 0.426, 0.944, 0.5835, 0.9673, 0.327, 0.457, 0.3103, 0.6304, 0.7285, 0.744, 0.919, 0.566, 0.9194, 0.522, 0.0554, 0.473, 0.22, 0.9546, 0.2241, 0.405, 0.3167, 0.0433, 0.9355, 0.3015, 0.94, 0.846, 0.205, 0.1771, 0.4512 },
        .{ 0.36, 0.1626, 0.3364, 0.953, 0.685, 0.6777, 0.811, 0.2345, 0.2964, 0.6763, 0.418, 0.817, 0.1467, 0.891, 0.4907, 0.329, 0.135, 0.3164, 0.8696, 0.2025, 0.716, 0.7583, 0.596, 0.01065, 0.527, 0.598, 0.3271, 0.729, 0.7554, 0.069, 0.2106, 0.587, 0.4106, 0.0245, 0.835, 0.4082, 0.0658 },
        .{ 0.5864, 0.05835, 0.498, 0.3982, 0.4895, 0.279, 0.6973, 0.775, 0.9243, 0.3274, 0.0955, 0.775, 0.8115, 0.4844, 0.3328, 0.1572, 0.8774, 0.1431, 0.4526, 0.932, 0.1986, 0.1859, 0.274, 0.6167, 0.0415, 0.755, 0.2494, 0.5684, 0.9033, 0.8657, 0.844, 0.8115, 0.05423, 0.5205, 0.9683, 0.3833, 0.0999 },
        .{ 0.3372, 0.3157, 0.1682, 0.617, 0.02077, 0.6772, 0.002575, 0.411, 0.8257, 0.7456, 0.5747, 0.095, 0.11957, 0.9736, 0.8745, 0.1744, 0.01607, 0.3645, 0.0811, 0.4172, 0.861, 0.2479, 0.1606, 0.8906, 0.351, 0.698, 0.912, 0.0955, 0.8438, 0.4797, 0.728, 0.6704, 0.5767, 0.2411, 0.657, 0.1243, 0.396 },
        .{ 0.763, 0.2201, 0.0913, 0.4229, 0.724, 0.6514, 0.5146, 0.854, 0.975, 0.9116, 0.3796, 0.927, 0.949, 0.517, 0.993, 0.10455, 0.2125, 0.5938, 0.7534, 0.11365, 0.574, 0.1508, 0.661, 0.4766, 0.6523, 0.5024, 0.03912, 0.4563, 0.8477, 0.684, 0.7695, 0.959, 0.982, 0.3896, 0.6685, 0.885, 0.0442 },
        .{ 0.3118, 0.3708, 0.975, 0.9766, 0.39, 0.589, 0.165, 0.6035, 0.922, 0.748, 0.9087, 0.288, 0.555, 0.583, 0.01729, 0.1309, 0.413, 0.978, 0.1962, 0.5513, 0.729, 0.6777, 0.3594, 0.873, 0.6377, 0.665, 0.381, 0.7505, 0.655, 0.794, 0.3123, 0.3257, 0.8823, 0.03174, 0.882, 0.4167, 0.988 },
        .{ 0.1388, 0.5127, 0.4927, 0.685, 0.6978, 0.509, 0.2201, 0.4084, 0.097, 0.3552, 0.2262, 0.29, 0.7563, 0.01544, 0.295, 0.66, 0.869, 0.608, 0.8022, 0.546, 0.7534, 0.8364, 0.4038, 0.01481, 0.737, 0.1826, 0.1831, 0.3613, 0.721, 0.6543, 0.1659, 0.6455, 0.613, 0.6035, 0.5513, 0.2301, 0.0358 },
        .{ 0.953, 0.495, 0.7427, 0.7466, 0.1289, 0.863, 0.9424, 0.3086, 0.3188, 0.03976, 0.4773, 0.0888, 0.2288, 0.457, 0.5386, 0.6387, 0.9927, 0.4004, 0.2275, 0.2091, 0.7695, 0.434, 0.1675, 0.4216, 0.7207, 0.1203, 0.301, 0.02495, 0.0471, 0.733, 0.2734, 0.2098, 0.3772, 0.05295, 0.9756, 0.1015, 0.5317 },
        .{ 0.2054, 0.03165, 0.4446, 0.54, 0.03152, 0.502, 0.6016, 0.5186, 0.57, 0.899, 0.4812, 0.4912, 0.343, 0.0798, 0.9873, 0.07074, 0.7695, 0.5884, 0.1692, 0.8564, 0.3347, 0.418, 0.5566, 0.4868, 0.571, 0.13, 0.7227, 0.811, 0.1034, 0.165, 0.532, 0.7456, 0.505, 0.2253, 0.1131, 0.6465, 0.864 },
        .{ 0.9126, 0.003994, 0.6265, 0.7593, 0.306, 0.6978, 0.836, 0.2756, 0.1534, 0.1208, 0.9307, 0.2773, 0.1687, 0.1473, 0.879, 0.00762, 0.9434, 0.5454, 0.5776, 0.9146, 0.575, 0.648, 0.5215, 0.9683, 0.1188, 0.0681, 0.01842, 0.7583, 0.3408, 0.1868, 0.588, 0.1011, 0.2578, 0.8535, 0.2401, 0.473, 0.534 },
    };
    const sample_Y: [29]bool = .{ true, false, false, true, false, true, true, false, false, true, true, true, false, false, true, true, true, false, false, true, true, false, true, false, true, false, false, true, true };
    test "Build the thing" {
        const samples = 29;
        const features = 37;
        // var gpa = std.testing.a(.{}){};
        // defer _ = gpa.deinit();
        const allocator = std.testing.allocator;
        const Model = model(features, samples);
        const my_model = Model{ .X = &sample_data, .Y = &sample_Y };
        const initial_params = try allocator.create([features]Precision);
        defer allocator.destroy(initial_params);
        const b: Precision = 0.1;
        for (0..features) |i| {
            initial_params[i] = 0.12344;
        }
        const prediction = try my_model.getPrediction(allocator, initial_params, b);
        defer prediction.deinit(allocator);

        const propagate = try my_model.propagate(allocator, initial_params, b);
        defer propagate.deinit(allocator);
        const opt = try my_model.optimize(allocator, 10000, 0.009, true);
        defer opt.deinit(allocator);
        std.debug.print("{d}\n", .{prediction.predictions});
    }
};

fn read_vec(comptime len: comptime_int, input: []const Precision) vectorType(len) {
    var vec: vectorType(len) = undefined;
    inline for (0..len) |i| {
        vec[i] = input[i];
    }
    return vec;
}

fn write_vec(comptime len: comptime_int, vals: vectorType(len), arr: []Precision) void {
    inline for (0..len) |i| {
        arr[i] = vals[i];
    }
}
