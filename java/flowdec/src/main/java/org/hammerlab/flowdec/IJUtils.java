package org.hammerlab.flowdec;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import ij.ImagePlus;
import ij.plugin.Concatenator;
import ij.process.FloatProcessor;

public class IJUtils {

	/**
	 * Convert 3d image (i.e. Z x X x Y x 1 image) to float array
	 */
	public static float[][][] toFloatArray(ImagePlus img) {
		int nz = img.getStackSize();
		float[][][] d = new float[nz][][];
		for (int z = 1; z <= nz; z++) {
			d[z-1] = img.getImageStack().getProcessor(z)
						.convertToFloat().getFloatArray();
		}
		return d;	
	}
	
	public static Map<String, Object> getImageStats(ImagePlus i) {
		Map<String, Object> s = new LinkedHashMap<>();
		s.put("Dims", Arrays.toString(i.getDimensions()));
		s.put("NFrames", i.getNFrames());
		s.put("NSlices", i.getNSlices());
		s.put("NChannels", i.getNChannels());
		s.put("DataType", i.getType());
		return s;
	}
	
	public static ImagePlus toImage(float[][][] data) {
		if (data.length == 0) {
			throw new IllegalArgumentException("Cannot create image from empty array");
		}
		ImagePlus[] stack = new ImagePlus[data.length];
		for (int z = 0; z < data.length; z++) {
			stack[z] = new ImagePlus("Slice " + z, new FloatProcessor(data[z]));
		}
		return new Concatenator().concatenate(stack, true);
	}

}
