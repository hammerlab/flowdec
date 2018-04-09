package org.hammerlab.flowdec;

import org.hammerlab.flowdec.FlowdecData.Acquisition;
import org.hammerlab.flowdec.FlowdecResults.TFDeconResult;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class FlowdecTest extends TestCase {

	public FlowdecTest(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(FlowdecTest.class);
	}

	/**
	 * Verify that graphs can be loaded
	 */
	public void testGraphLoad() {
		Flowdec.richardsonLucy()
			.task2d(new float[10][10], new float[10][10], 10)
			.processor();
	}

	/**
	 * Verify that graphs can also be run
	 */
	public void testGraphExecution() {
		Flowdec.richardsonLucy()
			.task2d(new float[10][10], new float[10][10], 10)
			.processor().run();
	}

	/**
	 * Test that execution of 2D project dataset deconvolution succeeds
	 */
	public void testDeconvolve2D() {
		Acquisition acq = FlowdecData.getDataset("bars-25pct");

		TFDeconResult res = Flowdec.richardsonLucy()
				.task2d(IJUtils.toFloatArray(acq.data)[0], 
						IJUtils.toFloatArray(acq.kernel)[0], 10).processor().call();

		res.data().float2d();
	}

	/**
	 * Test that execution of 3D project dataset deconvolution succeeds
	 */
	public void testDeconvolve3D() {
		Acquisition acq = FlowdecData.getDataset("bars-25pct");
		TFDeconResult res = Flowdec.richardsonLucy()
				.task3d(IJUtils.toFloatArray(acq.data), IJUtils.toFloatArray(acq.kernel), 10).processor().call();

		float[][][] data = res.data().float3d();
		IJUtils.toImage(data).show();
	}
}
