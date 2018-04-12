package org.hammerlab.flowdec;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Callable;

import org.hammerlab.flowdec.Flowdec.TensorResult;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Session.Runner;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;

import com.google.protobuf.InvalidProtocolBufferException;

public class FlowdecTask implements Callable<TensorResult>, Runnable {

	private Builder builder;
	private TensorResult result;
	
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
			this.model = SavedModelBundle.load(this.path.toString(), Flowdec.DEFAULT_SERVING_KEY);
			return new FlowdecTask(this);
		}
		
	}
	
	public static Builder newBuilder() {
		return new Builder();
	}
	
	private MetaGraphDef getMetaGraph() {
		try {
			return MetaGraphDef.parseFrom(this.builder.model.metaGraphDef());
		} catch (InvalidProtocolBufferException e) {
			throw new RuntimeException("Failed to retrieve meta graph from "
					+ "saved model at '" + this.builder.path + "'", e);
		}
	}
	
	
	@Override
	public synchronized void run() {

		if (this.result != null) {
			return;
		}
		
		Graph g = this.builder.model.graph();
		if (this.builder.device.isPresent()) {
			g = setDevice(g, this.builder.device.get());	
		}
		
		MetaGraphDef mg = this.getMetaGraph();
		SignatureDef sd = mg.getSignatureDefMap().get(Flowdec.DEFAULT_SERVING_SIGNATURE_DEF_KEY);
		
		Optional<ConfigProto> conf = this.builder.sessionConfig;
		try (Session sess = conf.isPresent() ? 
				new Session(g, conf.get().toByteArray()) : new Session(g)){
			
			Runner runner = sess.runner();
			
			Set<String> inputNames = sd.getInputsMap().keySet();
			Set<String> outputNames = sd.getOutputsMap().keySet();
			for (String name : this.builder.inputs.keySet()) {
				// Get name of tensor corresponding to input name; this is typically
				// similar to the name of the input itself but does need to be (eg
				// the input name may be "data" while the tensor placeholder is "data:0")
				if (!inputNames.contains(name)) {
					throw new IllegalArgumentException("Graph input '" + 
							name + "' not valid (valid inputs = '" + inputNames + ")");
				}
				String inputTensorName = sd.getInputsMap().get(name).getName();
				runner.feed(inputTensorName, this.builder.inputs.get(name));
			}
			for (String name : this.builder.outputs) {
				if (!outputNames.contains(name)) {
					throw new IllegalArgumentException("Graph output '" + 
							name + "' not valid (valid outputs = '" + outputNames + ")");
				}
				String outputTensorName = sd.getOutputsMap().get(name).getName();
				runner.fetch(outputTensorName);
			}

			Map<String, Tensor<?>> map = new LinkedHashMap<>();
			List<Tensor<?>> tensors = runner.run();
			for (int i = 0; i < this.builder.outputs.size(); i++) {
				map.put(this.builder.outputs.get(i), tensors.get(i));
			}
			this.result = new TensorResult(map);
		}
	}

	public synchronized TensorResult getResult() {
		return this.result;
	}
	
	@Override
	public synchronized TensorResult call() {
		this.run();
		return this.getResult();
	}
	
	/**
	 * Set device associated with TensorFlow graph execution
	 * 
	 * @param g TensorFlow graph instance
	 * @param device Name of device to place execution on; e.g. "/cpu:0", "/gpu:1", etc.
	 * See <a href="https://www.tensorflow.org/programmers_guide/faq">TF FAQ</a> for more details.
	 * @return Graph with all operations assigned to given device
	 */
	private static Graph setDevice(Graph g, String device) {
		
		// This functionality is based entirely on:
		// https://stackoverflow.com/questions/47799972/tensorflow-java-multi-gpu-inference

		GraphDef.Builder builder;
		try {
			builder = GraphDef.parseFrom(g.toGraphDef()).toBuilder();
		} catch (InvalidProtocolBufferException e) {
			throw new RuntimeException(e);
		}
		
		for (int i = 0; i < builder.getNodeCount(); ++i) {
			builder.getNodeBuilder(i).setDevice(device);
		}
		
		Graph gd = new Graph();
		gd.importGraphDef(builder.build().toByteArray());
		return gd;
	}
	
}
