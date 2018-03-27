package org.hammerlab.tfdecon;

import java.nio.file.Path;

import ij.IJ;
import ij.ImagePlus;

public class TFDeconData {

	public static class Acquisition {
    	public ImagePlus data;
    	public ImagePlus kernel;
    	public ImagePlus actual;
		public Acquisition(ImagePlus data, ImagePlus kernel, ImagePlus actual) {
			super();
			this.data = data;
			this.kernel = kernel;
			this.actual = actual;
		}	
    }
	
    public static Acquisition getDataset(String name) {
    	Path dataDir = TFDecon.getProjectDatasetDir().resolve(name);
    	ImagePlus data = IJ.openImage(dataDir.resolve("data.tif").toString());
    	ImagePlus kernel = IJ.openImage(dataDir.resolve("kernel.tif").toString());
    	ImagePlus actual = IJ.openImage(dataDir.resolve("actual.tif").toString());
    	return new Acquisition(data, kernel, actual);
    }
}
