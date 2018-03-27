package org.hammerlab.tfdecon;

import java.nio.file.Path;

import org.hammerlab.tfdecon.TFDeconData.Acquisition;
import org.hammerlab.tfdecon.TFDeconResults.TFDeconResult;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;


public class TFDeconTest extends TestCase {
	
    public TFDeconTest( String testName ) {
        super( testName );
    }
    
    public static Test suite() {
        return new TestSuite( TFDeconTest.class );
    }

    public void testGraphLoad() {
        TFDecon.richardsonLucy()
        	.task2d(new float[10][10], new float[10][10], 10)
        	.processor();
    }
    
    public void testGraphExecution() {
        TFDecon.richardsonLucy()
        	.task2d(new float[10][10], new float[10][10], 10)
        	.processor().run();
    }
    
    public void testDeconvolve2D() {
    	Acquisition acq = TFDeconData.getDataset("bars-25pct");
    	
    	System.out.println(IJUtils.getImageStats(acq.data));
    	System.out.println(IJUtils.getImageStats(acq.kernel));
    	
        TFDeconResult res = TFDecon.richardsonLucy()
    	.task2d(
    			IJUtils.toFloatArray(acq.data)[0], 
    			IJUtils.toFloatArray(acq.kernel)[0], 
    			10
    	).processor().call();
        
        res.data().float2d();
    }
    
    public void testDeconvolve3D() {
    	Acquisition acq = TFDeconData.getDataset("bars-25pct");
        TFDeconResult res = TFDecon.richardsonLucy()
    	.task3d(
			IJUtils.toFloatArray(acq.data), 
			IJUtils.toFloatArray(acq.kernel), 
			10
    	).processor().call();
        
        float[][][] data = res.data().float3d();
        IJUtils.toImage(data).show();
    }
}
