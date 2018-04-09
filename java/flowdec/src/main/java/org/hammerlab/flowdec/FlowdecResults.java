package org.hammerlab.flowdec;

import java.util.Map;

import org.tensorflow.Tensor;

public class FlowdecResults {

	public static final String DEFAULT_RESULT_TENSOR_KEY = "result";
	
	public static class TFDeconResult {
		
		private Map<String, Tensor<?>> data;
		
		public TFDeconResult(Map<String, Tensor<?>> data) {
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
