package org.hammerlab.flowdec;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Optional;

import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.framework.ConfigProto;

public class Flowdec {

	private static final String DEFAULT_PAD_MODE = "log2";
	
	static Path getProjectDir() {
		return Paths.get("..", "..").toAbsolutePath().normalize();
	}
	
	public static Path getProjectTensorflowDir() {
		return getProjectDir().resolve("tensorflow");
	}
	
	public static Path getProjectDatasetDir() {
		return getProjectDir().resolve("python/flowdec/datasets");
	}
	
	public static RichardsonLucy richardsonLucy() {
		return new RichardsonLucy();
	}
	
	public static enum ArgType {
		INPUT, OUTPUT
	}
	
	public static enum Arg {

		DATA("data", ArgType.INPUT),
		KERNEL("kernel", ArgType.INPUT),
		NITER("niter", ArgType.INPUT),
		RESULT("result", ArgType.OUTPUT);
		
		public final String name;
		public final ArgType type;
		
		Arg(String name, ArgType type){
			this.name = name;
			this.type = type;
		}
	}

	
	@SuppressWarnings("rawtypes")
	public static abstract class Algo<T extends Algo> {
		
		protected Optional<Path> path = Optional.empty();
		protected Optional<String> device = Optional.empty();
		protected Optional<ConfigProto> sessConfig = Optional.empty();
		
		Algo() {}
		
		@SuppressWarnings("unchecked")
		public T setModelPath(Path path) {
			this.path = Optional.ofNullable(path);
			return (T) this;
		}
		
		@SuppressWarnings("unchecked")
		public T setDevice(String device) {
			this.device = Optional.ofNullable(device);
			return (T) this;
		}
		
		@SuppressWarnings("unchecked")
		public T setSessConfig(ConfigProto sessConfig) {
			this.sessConfig = Optional.ofNullable(sessConfig);
			return (T) this;
		}
		
	}
	
	public static class RichardsonLucy extends Algo<RichardsonLucy> {
		
		private static final String DOMAIN_TYPE = "complex";
		
		public FlowdecTask task2d(float[][] data, float[][] kernel, int niter) {
			return this.task(Tensors.create(data), Tensors.create(kernel), niter);
		}
		
		public FlowdecTask task3d(float[][][] data, float[][][] kernel, int niter) {
			return this.task(Tensors.create(data), Tensors.create(kernel), niter);
		}
		
		public FlowdecTask task(Tensor<?> data, Tensor<?> kernel, int niter) {
			
			if (data.shape().length != kernel.shape().length) {
				throw new IllegalArgumentException(String.format(
					"Data and kernel must have same number of "
					+ "dimensions (data shape = %s, kernel shape = %s)",
					Arrays.toString(data.shape()), Arrays.toString(kernel.shape())
				));
			}
			int ndims = data.shape().length;
			
			FlowdecTask.Builder builder = FlowdecTask.newBuilder()
				.addInput("data", data)
				.addInput("kernel", kernel)
				.addInput("niter", Tensors.create(niter))
				.addInput("pad_mode", Tensors.create(DEFAULT_PAD_MODE))
				.addOutput("result");
			
			if (this.path.isPresent()) {
				builder = builder.setModelPath(this.path.get());
			} else {
				Path modelPath = Flowdec.getProjectTensorflowDir()
						.resolve("richardson-lucy-" + DOMAIN_TYPE + "-" + ndims + "d")
						.normalize()
						.toAbsolutePath();
				builder = builder.setModelPath(modelPath);
			}
			
			if (this.device.isPresent())
				builder = builder.setDevice(this.device.get());
			
			if (this.sessConfig.isPresent())
				builder = builder.setSessionConfig(this.sessConfig.get());
			
			return builder.build();
		}


	}
	
}
