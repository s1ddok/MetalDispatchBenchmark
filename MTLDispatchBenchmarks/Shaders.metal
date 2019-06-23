//
//  Shaders.metal
//  MTLDispatchBenchmarks
//
//  Created by Andrey Volodin on 13/06/2019.
//  Copyright © 2019 Andrey Volodin. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

inline void increment(texture2d<int, access::read> input,
                      texture2d<int, access::write> output,
                      ushort2 coordinate) {
    const int4 current = input.read(coordinate);

    output.write((current + 1) % 1000, coordinate);
}


kernel void uniformIncrement(texture2d<int, access::read> input,
                             texture2d<int, access::write> output,
                             ushort2 coordinate [[thread_position_in_grid]]) {
    if (coordinate.x >= input.get_width() || coordinate.y >= input.get_height()) {
        return;
    }

    increment(input, output, coordinate);
}

kernel void exactIncrement(texture2d<int, access::read> input,
                           texture2d<int, access::write> output,
                           ushort2 coordinate [[thread_position_in_grid]]) {
    increment(input, output, coordinate);
}


