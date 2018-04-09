package org.hammerlab.flowdec;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;

public class FlowdecTask {

	private static final String DEFAULT_SERVING_KEY = "serve";
		
	Builder builder;
	
	FlowdecTask(Builder builder){
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
			this.addInput(Flowdec.Arg.DATA.name, data);
			this.addInput(Flowdec.Arg.KERNEL.name, kernel);
			this.addInput(Flowdec.Arg.NITER.name, niter);
			this.addOutput(Flowdec.Arg.RESULT.name);
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
		
		public FlowdecTask build() {
			this.model = SavedModelBundle.load(this.path.toString(), DEFAULT_SERVING_KEY);
			return new FlowdecTask(this);
		}
		
	}
	
	public static Builder newBuilder() {
		return new Builder();
	}
	
	public FlowdecProcessor processor() {
		return new FlowdecProcessor(this);
	}
	
}
