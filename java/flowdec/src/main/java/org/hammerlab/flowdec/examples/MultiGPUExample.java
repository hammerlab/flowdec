package org.hammerlab.flowdec.examples;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.hammerlab.flowdec.FlowdecTask;
import org.hammerlab.flowdec.IJUtils;
import org.tensorflow.Tensors;
import org.tensorflow.framework.ConfigProto;

import ij.IJ;

/**
 * Example deconvolution script used to process an arbitrary number
 * of acquisitions under the assumption of the same PSF on multiple
 * GPUs (if present).
 * 
 * @author eczech
 *
 */
public class MultiGPUExample {

	public static void main(String[] args) {

		// Path containing images to deconvolve
		Path imgDir = Paths.get(args[0]);

		// Path to single PSF applying to all images
		Path psfPath = Paths.get(args[1]);
		
		// Number of deconvolution iterations
		int nIterations = 50;
		
		// Number of gpus present on system
		int numGpus = 2;

		// Open all images and psf as 3D float arrays
		List<float[][][]> imgs = Arrays.asList(imgDir.toFile().list())
				.stream().map(MultiGPUExample::open)
				.collect(Collectors.toList());
		float[][][] psf = open(psfPath.toString());

		// Set properties for tensorflow session (e.g. logging device placement,
		// hard v soft device settings, parallelism, etc.)
		// See: https://www.tensorflow.org/api_docs/python/tf/Session
		ConfigProto config = ConfigProto.newBuilder().build();
		
		// Build (callable/runnable) deconvolution tasks for each image,
		// assign each a GPU, run the processing, and then fetch results
		// as 3d float arrays.
		// Note: Control parallelism by wrapping in fork pool or via:
		// -Djava.util.concurrent.ForkJoinPool.common.parallelism=NUM_GPUS
		AtomicInteger ct = new AtomicInteger();
		List<float[][][]> results = imgs.parallelStream()
			.map(img -> 
				FlowdecTask.newBuilder()
				.setSessionConfig(config)
				.setArgs(
					Tensors.create(img), 
					Tensors.create(psf), 
					Tensors.create(nIterations)
				)
				.setDevice("/gpu:" + (ct.incrementAndGet() % numGpus))
				.build().call()
				.data().float3d()
			).collect(Collectors.toList());
		
		// Do something with the results
		System.out.println("Finished deconvolution of " + results.size() + " images");
	}
	
	private static float[][][] open(String img) {
		return IJUtils.toFloatArray(IJ.openImage(img));
	}
}
