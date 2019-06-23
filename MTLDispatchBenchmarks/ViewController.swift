//
//  ViewController.swift
//  MTLDispatchBenchmarks
//
//  Created by Andrey Volodin on 13/06/2019.
//  Copyright © 2019 Andrey Volodin. All rights reserved.
//

import UIKit
import Alloy

extension MTLSize: Hashable, CustomDebugStringConvertible {
    public static func == (lhs: MTLSize, rhs: MTLSize) -> Bool {
        return lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.width)
        hasher.combine(self.height)
        hasher.combine(self.depth)
    }
    
    public var debugDescription: String { "\(self.width), \(self.height), \(self.depth)" }
}

struct Configuration: Hashable {
    var threadgroupSize: MTLSize
    var pipelineState: String
}

class ViewController: UIViewController {

    var context: MTLContext!

    var exactState: MTLComputePipelineState!
    var uniformState: MTLComputePipelineState!
    var optimizedUniformState: MTLComputePipelineState!

    var texture1: MTLTexture!
    var texture2: MTLTexture!

    override func viewDidLoad() {
        super.viewDidLoad()

        let context = MTLContext(device: Metal.device)

        self.context = context
        
        guard
            let library = context.standardLibrary,
            let exactPipelineState = try? library.computePipelineState(function: "exactIncrement"),
            let uniformPipelineState = try? library.computePipelineState(function: "uniformIncrement")
        else {
            fatalError()
        }

        let kernelDescriptor = MTLComputePipelineDescriptor()
        kernelDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

        let function = library.makeFunction(name: "exactIncrement")

        kernelDescriptor.computeFunction = function

        guard let uniformFusedPipelineState = try? context.device
                                                          .makeComputePipelineState(descriptor: kernelDescriptor,
                                                                                    options: [],
                                                                                    reflection: nil)
        else {
            fatalError()
        }


        let texture1 = context.texture(width: 513, height: 513, pixelFormat: .rgba32Sint, usage: [.shaderRead, .shaderWrite])
        let texture2 = context.texture(width: 513, height: 513, pixelFormat: .rgba32Sint, usage: [.shaderRead, .shaderWrite])

        self.texture1 = texture1
        self.texture2 = texture2

        self.exactState = exactPipelineState
        self.uniformState = uniformPipelineState
        self.optimizedUniformState = uniformFusedPipelineState

        print(self.exactState.threadExecutionWidth)
        print(self.uniformState.threadExecutionWidth)
        print(self.optimizedUniformState.threadExecutionWidth)
    }

    var results: [(TimeInterval, Configuration)] = []

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(true)

        let iterations = 400
        
        results.removeAll()
        for _ in 0...20 {
            let possibleDivisors = (0...5).map { Int(pow(2.0, Double($0))) }.shuffled()
            
            for d in possibleDivisors {
                var dispatchGroupSize = self.uniformState.max2dThreadgroupSize
                dispatchGroupSize.height /= d
                self.testDispatch(configuration: Configuration(threadgroupSize: dispatchGroupSize,
                                                               pipelineState: "uniform"),
                                  iterations: iterations)
                /*self.testDispatch(configuration: Configuration(threadgroupSize: dispatchGroupSize,
                                                               pipelineState: "uniformOptimized"),
                                  iterations: iterations)*/
                self.testDispatch(configuration: Configuration(threadgroupSize: dispatchGroupSize,
                                                               pipelineState: "exact"),
                                  iterations: iterations)
            }
        }

        //try? context.schedule { buffer in
        //    buffer.addCompletedHandler { _ in
                print(self.results.map { $0.0 })
                
                print("---- TIME END ---")
                
                let sortedResults = self.results.sorted { $0.0 < $1.0 }

                let normalizedResults = sortedResults.map { ($0.0 / sortedResults.first!.0, $0.1.pipelineState, $0.1.threadgroupSize) }
                print(normalizedResults)
                
                print("--- GLOBAL NORMALIZED END")
                let groupedByThreadgroup = Dictionary(grouping: self.results) { $0.1.threadgroupSize }
            
                for (key, results) in groupedByThreadgroup {
                    print("Best results for key: \(key)")
                    print(results.sorted { $0.0 < $1.0 }.map { ($0.0, $0.1.pipelineState) }[0..<3])
                }
         //   }
        //}
    }

    func testDispatch(configuration: Configuration, iterations: Int) {
        try? self.context.scheduleAndWait { buffer in
            buffer.compute { encoder in
                for i in 1...iterations {
                    let t1 = i % 2 == 0 ? self.texture1 : self.texture2
                    let t2 = i % 2 == 0 ? self.texture2 : self.texture1

                    encoder.set(textures: [t1, t2])

                    let state: MTLComputePipelineState

                    switch configuration.pipelineState {
                    case "uniform": state = self.uniformState
                    case "uniformOptimized": state = self.optimizedUniformState
                    case "exact": state = self.exactState
                    default: fatalError()
                    }

                    if configuration.pipelineState != "exact" {
                        encoder.dispatch2d(state: state,
                                           covering: t1!.size,
                                           threadgroupSize: configuration.threadgroupSize)
                    } else {
                        encoder.dispatch2d(state: state,
                                           exactly: t1!.size,
                                           threadgroupSize: configuration.threadgroupSize)
                    }
                }

                buffer.addCompletedHandler { _buffer in
                    self.results.append((_buffer.gpuExecutionTime, configuration))
                }
            }
        }
    }


}

