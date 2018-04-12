package org.hammerlab.flowdec;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;

import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.framework.ConfigProto;

public class Flowdec {

	static final String DEFAULT_RESULT_TENSOR_KEY = "result";
	
	// See: https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/saved_model/signature_constants.py
	static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";
	
	static final String DEFAULT_SERVING_KEY = "serve";
	
	static Path getProjectDir() {
		return Paths.get("..", "..").toAbsolutePath().normalize();
	}
	
	public static Path getProjectTensorFlowDir() {
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
				.addOutput("result");
			
			if (this.path.isPresent()) {
				builder = builder.setModelPath(this.path.get());
			} else {
				Path modelPath = Flowdec.getProjectTensorFlowDir()
						.resolve("richardsonlucy-" + DOMAIN_TYPE + "-" + ndims + "d")
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
	
	
	public static class TensorResult {
		
		private Map<String, Tensor<?>> data;
		
		public TensorResult(Map<String, Tensor<?>> data) {
			this.data = data;
		}
		
		public Tensor<?> getTensor(String name){
			if (this.data.isEmpty()) {
				throw new IllegalStateException(
						"No data found in TF graph results");
			}
			if (!this.data.containsKey(name)) {
				throw new IllegalStateException("Failed to find result '" + 
						name + "' in TF Graph results");
			}
			return this.data.get(name);
		}
		
		public TensorData data() {
			return this.data(DEFAULT_RESULT_TENSOR_KEY);
		}
		
		public TensorData data(String tensor) {
			return new TensorData(this.getTensor(tensor));
		}
		
	}
	
	public static class TensorData {
		private final Tensor<?> data;

		public TensorData(Tensor<?> data) {
			super();
			this.data = data;
		}

		protected float[][] float2d() {
			Tensor<Float> res = this.data.expect(Float.class);
			if (res.shape().length != 2) {
				throw new IllegalStateException("Tensor result has " + 
						res.shape().length + " dimensions but exactly 2 were expected");
			}
			int x = (int) res.shape()[0];
			int y = (int) res.shape()[1];
			float[][] arr = new float[x][y];
			res.copyTo(arr);
			return arr;
		}
		
		public float[][][] float3d() {
			Tensor<Float> res = this.data.expect(Float.class);
			if (res.shape().length != 3) {
				throw new IllegalStateException("Tensor result has " + 
						res.shape().length + " dimensions but exactly 3 were expected");
			}
			int z = (int) res.shape()[0];
			int x = (int) res.shape()[1];
			int y = (int) res.shape()[2];
			float[][][] arr = new float[z][x][y];
			res.copyTo(arr);
			return arr;
		}
		
	}
	
}
