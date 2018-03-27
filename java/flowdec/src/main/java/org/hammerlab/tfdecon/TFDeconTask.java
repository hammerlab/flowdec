package org.hammerlab.tfdecon;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;

public class TFDeconTask {

	private static final String DEFAULT_SERVING_KEY = "serve";
		
	Builder builder;
	
	TFDeconTask(Builder builder){
		this.builder = builder;
	}
	
	public static class Builder {
		Path path;
		Map<String, Tensor<?>> inputs;
		List<String> outputs;
		Optional<String> device;
		Optional<ConfigProto> sessionConfig;
		SavedModelBundle model;
		
		public Builder() {
			this.inputs = new LinkedHashMap<>();
			this.outputs = new LinkedList<>();
			this.device = Optional.empty();
			this.sessionConfig = Optional.empty();
		}
		
		public Builder setModelPath(Path path) {
			this.path = path;
			return this;
		}
		
		public Builder setArgs(Tensor<?> data, Tensor<?> kernel, Tensor<?> niter) {
			this.addInput(TFDecon.Arg.DATA.name, data);
			this.addInput(TFDecon.Arg.KERNEL.name, kernel);
			this.addInput(TFDecon.Arg.NITER.name, niter);
			this.addOutput(TFDecon.Arg.RESULT.name);
			return this;
		}
		
		Builder addInput(String name, Tensor<?> value) {
			this.inputs.put(name, value);
			return this;
		}
		
		Builder addOutput(String name) {
			this.outputs.add(name);
			return this;
		}
		
		public Builder setDevice(String device) {
			this.device = Optional.ofNullable(device);
			return this;
		}
		
		public Builder setSessionConfig(ConfigProto config) {
			this.sessionConfig = Optional.ofNullable(config);
			return this;
		}
		
		public TFDeconTask build() {
			this.model = SavedModelBundle.load(this.path.toString(), DEFAULT_SERVING_KEY);
			return new TFDeconTask(this);
		}
		
	}
	
	public static Builder newBuilder() {
		return new Builder();
	}
	
	public TFDeconProcessor processor() {
		return new TFDeconProcessor(this);
	}
	
}
