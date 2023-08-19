import tensorflow as tf

class FCQ(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=tf.nn.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = tf.keras.layers.Dense(hidden_dims[0])

        self.hidden_layers = []
        for i in range(len(hidden_dims)-1):
            hidden_layer = tf.keras.layers.Dense(hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            #x = tf.expand_dims(x, axis=0)
        return x

    def call(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

    def numpy_float_to_device(self, variable):
        variable = tf.convert_to_tensor(variable, dtype=tf.float32)
        return variable

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        is_terminals = tf.convert_to_tensor(is_terminals, dtype=tf.float32)
        return states, actions, rewards, new_states, is_terminals

    def plot_model(self):
        x = tf.keras.layers.Input(shape = (self.input_dim,))
        model = tf.keras.models.Model(inputs=[x], outputs=[self.call(x)])
        tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
