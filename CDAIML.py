from matplotlib import pyplot
from math import cos, sin, atan, pi
from palettable.tableau import Tableau_10
from time import localtime, strftime
import numpy as np

from IPython.display import HTML, display
import urllib.parse

# This Library is modified based on the work by Milo Spencer-Harper and Oli Blum, https://stackoverflow.com/a/37366154/10404826
# On top of that, I added support for showing weights (linewidth, colors, etc.)
# Contributor: Jianzheng Liu
# Contact: jzliu.100@gmail.com


# --- Configuration Defaults ---
# These can be overridden by passing arguments to DrawNN
DEFAULT_NEURON_RADIUS = 0.5
DEFAULT_NEURON_SPACING_HORIZONTAL = 2.0
DEFAULT_LAYER_SPACING_VERTICAL = 6.0
DEFAULT_NEURON_ID_Y_OFFSET = 0.15  # Offset for neuron ID text from neuron center
DEFAULT_NEURON_ID_FONTSIZE = 10

DEFAULT_LINEWIDTH_BASE = 0.2  # Minimum linewidth for small weights
DEFAULT_LINEWIDTH_SCALE_TIER1 = 5.0  # Multiplier for weights > threshold1
DEFAULT_LINEWIDTH_SCALE_TIER2 = 10.0 # Multiplier for weights > threshold2
DEFAULT_WEIGHT_THRESHOLD_TIER1 = 0.5
DEFAULT_WEIGHT_THRESHOLD_TIER2 = 0.8
DEFAULT_MAX_LINEWIDTH = 4.0

DEFAULT_WEIGHT_TEXT_SHOW_THRESHOLD = 0.5 # Only show text for abs(weights) > this
DEFAULT_WEIGHT_TEXT_FONTSIZE = 8
DEFAULT_WEIGHT_TEXT_PRECISION = "{:3.2f}"
DEFAULT_WEIGHT_TEXT_BBOX_ALPHA = 0.0 # Bbox alpha for weight text (0 for transparent)
DEFAULT_WEIGHT_TEXT_BBOX_WIDTH = 0.8 # Estimated width for text overlap check
DEFAULT_WEIGHT_TEXT_BBOX_HEIGHT = 0.4 # Estimated height for text overlap check

DEFAULT_TEXT_OVERLAP_GRID_SIZE = 0.2
DEFAULT_TEXT_LABEL_SEARCH_SEGMENTS = 10 # Number of segments on line to check for text placement
DEFAULT_TEXT_LABEL_SEARCH_START_OFFSET_DIVISOR = 2 # e.g., 2 means start search at middle

DEFAULT_LAYER_LABEL_FONTSIZE = 12
DEFAULT_FIGSIZE = (12, 9)
DEFAULT_OUTPUT_FILENAME_PREFIX = 'ANN_'
DEFAULT_DPI = 300
DEFAULT_TITLE_FONTSIZE = 15

class Neuron():
    """
    Represents a single neuron in the neural network visualization.
    """
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.config = config

    def draw(self, neuron_id=-1):
        """
        Draws the neuron as a circle and its ID.
        """
        circle = pyplot.Circle(
            (self.x, self.y),
            radius=self.config['neuron_radius'],
            fill=False
        )
        pyplot.gca().add_patch(circle)
        if neuron_id != -1:
            pyplot.gca().text(
                self.x,
                self.y - self.config['neuron_id_y_offset'] - self.config['neuron_radius'], # Adjust based on radius
                str(neuron_id),
                size=self.config['neuron_id_fontsize'],
                ha='center',
                va='top' # Align to top to be below neuron
            )

class Layer():
    """
    Represents a layer of neurons in the neural network visualization.
    """
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, config):
        self.config = config
        self.neuron_radius = self.config['neuron_radius']
        self.horizontal_distance_between_neurons = self.config['neuron_spacing_horizontal']
        self.vertical_distance_between_layers = self.config['layer_spacing_vertical']
        
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self._get_previous_layer(network)
        self.y = self._calculate_layer_y_position()
        self.neurons = self._initialise_neurons(number_of_neurons)

    def _initialise_neurons(self, number_of_neurons):
        """
        Creates and positions the neurons for this layer.
        """
        neurons = []
        x = self._calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for _ in range(number_of_neurons):
            neuron = Neuron(x, self.y, self.config)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def _calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        """
        Calculates the x-offset to center the current layer horizontally.
        """
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2.0

    def _calculate_layer_y_position(self):
        """
        Calculates the y-coordinate for this layer.
        """
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0 # First layer is at y=0

    def _get_previous_layer(self, network):
        """
        Returns the previous layer in the network, if any.
        """
        return network.layers[-1] if network.layers else None

    def _line_between_two_neurons(self, neuron1, neuron2, weight, textoverlaphandler):
        """
        Draws a line (representing a weight) between two neurons.
        Also attempts to place the weight value as text if significant.
        """
        # Angle and adjustments for line start/end at neuron circumference
        delta_y = neuron2.y - neuron1.y
        if abs(delta_y) < 1e-6: # Avoid division by zero if neurons are at same y (should not happen in layered network)
            angle = pi / 2 if neuron2.x > neuron1.x else -pi / 2
        else:
            angle = atan((neuron2.x - neuron1.x) / delta_y)

        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        # Determine line color based on weight sign
        color = Tableau_10.mpl_colors[0]  # Default (e.g., negative or zero)
        if weight > 0:
            color = Tableau_10.mpl_colors[1] # Positive

        # Determine linewidth based on weight magnitude
        abs_weight = abs(weight)
        if abs_weight > self.config['weight_threshold_tier2']:
            linewidth = self.config['linewidth_scale_tier2'] * abs_weight
        elif abs_weight > self.config['weight_threshold_tier1']:
            linewidth = self.config['linewidth_scale_tier1'] * abs_weight
        else:
            linewidth = self.config['linewidth_base']
        
        linewidth = min(linewidth, self.config['max_linewidth']) # Cap linewidth

        # Draw the line
        line = pyplot.Line2D(
            (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
            linewidth=linewidth,
            color=color,
            zorder=1 # Ensure lines are behind neurons and text if needed
        )
        pyplot.gca().add_line(line)

        # Draw weight text if significant and space allows
        if abs_weight > self.config['weight_text_show_threshold'] and textoverlaphandler:
            num_segments = self.config['text_label_search_segments']
            start_search_offset = num_segments // self.config['text_label_search_start_offset_divisor']
            
            # Points on the line segment (excluding ends covered by neurons)
            line_x_start = neuron1.x - x_adjustment
            line_y_start = neuron1.y - y_adjustment
            line_x_end = neuron2.x + x_adjustment
            line_y_end = neuron2.y + y_adjustment

            placed_text = False
            # Try placing text starting from middle of the line and expanding outwards
            for i in range(num_segments // 2 + 1):
                if placed_text: break
                # Try position offset towards neuron2
                s1 = start_search_offset + i
                if 0 < s1 < num_segments: # Ensure step is within the line segment parts
                    t = s1 / float(num_segments)
                    txt_x_pos = line_x_start + t * (line_x_end - line_x_start)
                    txt_y_pos = line_y_start + t * (line_y_end - line_y_start)
                    
                    # Define bounding box for text
                    text_bbox_width = self.config['weight_text_bbox_width']
                    text_bbox_height = self.config['weight_text_bbox_height']
                    coords = [
                        txt_x_pos - text_bbox_width / 2, txt_y_pos - text_bbox_height / 2,
                        txt_x_pos + text_bbox_width / 2, txt_y_pos + text_bbox_height / 2
                    ]
                    if textoverlaphandler.getspace(coords):
                        text_obj = pyplot.gca().text(
                            txt_x_pos, txt_y_pos,
                            self.config['weight_text_precision'].format(weight),
                            size=self.config['weight_text_fontsize'],
                            ha='center', va='center', zorder=3 # Text above lines/neurons
                        )
                        text_obj.set_bbox(dict(facecolor='white', alpha=self.config['weight_text_bbox_alpha'], edgecolor='none', pad=0.1))
                        placed_text = True
                        break # Found a spot

                if placed_text: break
                # Try position offset towards neuron1 (if different from s1)
                s2 = start_search_offset - i
                if i != 0 and 0 < s2 < num_segments : # Ensure step is valid and not redundant
                    t = s2 / float(num_segments)
                    txt_x_pos = line_x_start + t * (line_x_end - line_x_start)
                    txt_y_pos = line_y_start + t * (line_y_end - line_y_start)
                    coords = [ # Re-calculate coords with potentially new txt_x_pos, txt_y_pos
                        txt_x_pos - text_bbox_width / 2, txt_y_pos - text_bbox_height / 2,
                        txt_x_pos + text_bbox_width / 2, txt_y_pos + text_bbox_height / 2
                    ]
                    if textoverlaphandler.getspace(coords):
                        text_obj = pyplot.gca().text(
                            txt_x_pos, txt_y_pos,
                            self.config['weight_text_precision'].format(weight),
                            size=self.config['weight_text_fontsize'],
                            ha='center', va='center', zorder=3
                        )
                        text_obj.set_bbox(dict(facecolor='white', alpha=self.config['weight_text_bbox_alpha'], edgecolor='none', pad=0.1))
                        placed_text = True
                        break # Found a spot


    def draw(self, layer_type_label, weights_matrix=None, textoverlaphandler=None):
        """
        Draws all neurons in the layer and connections to the previous layer.
        layer_type_label: String like "Input Layer", "Hidden Layer 1", "Output Layer".
        weights_matrix: A 2D numpy array of weights connecting previous layer to this layer.
                        Shape: (num_neurons_previous_layer, num_neurons_current_layer)
        """
        for j, neuron in enumerate(self.neurons): # j is index for neurons in this layer
            neuron.draw(neuron_id=j + 1) # Draw neuron (ID is 1-based)
            if self.previous_layer and weights_matrix is not None:
                for i, previous_layer_neuron in enumerate(self.previous_layer.neurons): # i is index for neurons in previous layer
                    weight = weights_matrix[i, j]
                    self._line_between_two_neurons(neuron, previous_layer_neuron, weight, textoverlaphandler)
        
        # Layer Text Label
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons + self.config['neuron_spacing_horizontal'] # Position to the right
        pyplot.text(x_text, self.y, layer_type_label, fontsize=self.config['layer_label_fontsize'], va='center')


class TextOverlappingHandler():
    """
    Handles text overlapping by discretizing the plot area into a grid
    and marking cells as occupied.
    """
    def __init__(self, plot_width, plot_height, grid_size):
        self.grid_size = grid_size
        self.plot_width = plot_width
        self.plot_height = plot_height

        # Calculate grid dimensions, ensuring they cover the area
        # Add a small buffer to plot_width/height in case coordinates are right on the edge
        num_x_cells = int(np.ceil((plot_width + 1e-9) / grid_size))
        num_y_cells = int(np.ceil((plot_height + 1e-9) / grid_size))
        
        self.cells = np.ones((num_x_cells, num_y_cells), dtype=bool)

    def getspace(self, test_coordinates):
        """
        Checks if the rectangular space defined by test_coordinates is available.
        If available, marks the corresponding grid cells as occupied and returns True.
        Otherwise, returns False.
        test_coordinates: [x_min, y_min, x_max, y_max] for the text bounding box.
        """
        # Ensure coordinates are within the handler's managed area by clipping
        # (assuming 0,0 is bottom-left of the managed area)
        x_min = max(0, test_coordinates[0])
        y_min = max(0, test_coordinates[1])
        x_max = min(self.plot_width, test_coordinates[2])
        y_max = min(self.plot_height, test_coordinates[3])

        if x_min >= x_max or y_min >= y_max: # Invalid or zero-size box after clipping
            return False

        # Convert to grid indices
        # Epsilon added to x_max/y_max to correctly include cells if the edge falls exactly on a grid line.
        # A cell (i,j) covers area [i*gs, (i+1)*gs) x [j*gs, (j+1)*gs)
        x_start_idx = int(np.floor(x_min / self.grid_size))
        y_start_idx = int(np.floor(y_min / self.grid_size))
        x_end_idx = int(np.floor((x_max - 1e-9) / self.grid_size)) # -epsilon to keep it in cell if on boundary
        y_end_idx = int(np.floor((y_max - 1e-9) / self.grid_size))


        max_x_grid_idx, max_y_grid_idx = self.cells.shape
        
        # Boundary checks for indices
        x_start_idx = max(0, min(x_start_idx, max_x_grid_idx - 1))
        y_start_idx = max(0, min(y_start_idx, max_y_grid_idx - 1))
        x_end_idx = max(0, min(x_end_idx, max_x_grid_idx - 1))
        y_end_idx = max(0, min(y_end_idx, max_y_grid_idx - 1))

        # Check if all cells in the rectangle are available
        for i in range(x_start_idx, x_end_idx + 1):
            for j in range(y_start_idx, y_end_idx + 1):
                if not self.cells[i, j]:
                    return False  # Space is already occupied

        # If all cells are free, mark them as occupied
        for i in range(x_start_idx, x_end_idx + 1):
            for j in range(y_start_idx, y_end_idx + 1):
                self.cells[i, j] = False
        return True

class NeuralNetwork():
    """
    Manages the collection of layers and orchestrates the drawing process.
    """
    def __init__(self, number_of_neurons_in_widest_layer, config):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.config = config

    def add_layer(self, number_of_neurons):
        """
        Adds a new layer to the network.
        """
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, self.config)
        self.layers.append(layer)

    def draw(self, weights_list=None):
        """
        Draws the entire neural network.
        weights_list: A list of 2D numpy arrays, where weights_list[k] contains
                      weights between layer k and layer k+1.
        """
        # Calculate total width and height for TextOverlappingHandler initialization
        # Network width considers neuron positions only. Text labels for layers might extend further.
        network_plot_width = (self.number_of_neurons_in_widest_layer -1) * self.config['neuron_spacing_horizontal'] + \
                             2 * self.config['neuron_radius'] 
        if self.number_of_neurons_in_widest_layer == 1: # Special case for single neuron per layer
            network_plot_width = 2 * self.config['neuron_radius']
        
        # Network height for y-positions from 0 up to the last layer's y-position.
        network_plot_height = (len(self.layers) - 1) * self.config['layer_spacing_vertical'] + \
                              2 * self.config['neuron_radius']
        if len(self.layers) == 1:
            network_plot_height = 2 * self.config['neuron_radius']
            
        # The actual plot area for text might be larger if labels are outside the core neuron area.
        # For simplicity, we use a slightly expanded area for the text handler.
        # Or, more accurately, it should cover where text can appear.
        # The layer labels are outside this `network_plot_width`. Max layer label could be a factor.
        # For now, using these calculated values. May need adjustment if text clipping occurs.
        text_handler_width = network_plot_width + 3 * self.config['neuron_spacing_horizontal'] # Extra space for layer labels
        text_handler_height = network_plot_height
        
        overlaphandler = TextOverlappingHandler(
            text_handler_width,
            text_handler_height,
            grid_size=self.config['text_overlap_grid_size']
        )

        fig = pyplot.figure(figsize=self.config['figsize'])
        
        for i, layer in enumerate(self.layers):
            layer_label = ""
            current_weights = None
            if i == 0:
                layer_label = 'Input Layer'
            elif i == len(self.layers) - 1:
                layer_label = 'Output Layer'
                if weights_list and len(weights_list) > i - 1:
                    current_weights = weights_list[i-1]
            else:
                layer_label = f'Hidden Layer {i}'
                if weights_list and len(weights_list) > i - 1:
                    current_weights = weights_list[i-1]
            
            layer.draw(layer_type_label=layer_label, weights_matrix=current_weights, textoverlaphandler=overlaphandler)

        pyplot.axis('scaled')
        pyplot.axis('off')
        plot_title = self.config.get('title', 'Neural Network Architecture') # Allow custom title
        pyplot.title(plot_title, fontsize=self.config['title_fontsize'])
        
        #filename_prefix = self.config['output_filename_prefix']
        #timestamp = strftime("%Y%m%d_%H%M%S", localtime())
        #figure_name = f'{filename_prefix}{timestamp}.png'
        
        #pyplot.savefig(figure_name, dpi=self.config['dpi'], bbox_inches="tight")
        #print(f"Neural network visualization saved as {figure_name}")
        pyplot.show()
        pyplot.close(fig) # Close the figure to free memory


class DrawNN():
    """
    Main class to instantiate and draw a neural network visualization.
    """
    def __init__(self, neural_network_layers, weights_list=None, **kwargs):
        """
        Initializes the neural network visualizer.

        Args:
            neural_network_layers (list of int): Number of neurons in each layer,
                                                 e.g., [input_size, hidden1_size, ..., output_size].
            weights_list (list of np.ndarray, optional): List of weight matrices.
                                                         weights_list[i] connects layer i to layer i+1.
                                                         Shape of weights_list[i] should be
                                                         (neurons_in_layer_i, neurons_in_layer_i+1).
                                                         If None, default weights are used for visualization.
            **kwargs: Additional configuration parameters to override defaults. See DEFAULT_* constants.
        """
        self.neural_network_layers = neural_network_layers
        self.weights_list = weights_list

        # --- Setup Configuration ---
        self.config = {
            'neuron_radius': kwargs.get('neuron_radius', DEFAULT_NEURON_RADIUS),
            'neuron_spacing_horizontal': kwargs.get('neuron_spacing_horizontal', DEFAULT_NEURON_SPACING_HORIZONTAL),
            'layer_spacing_vertical': kwargs.get('layer_spacing_vertical', DEFAULT_LAYER_SPACING_VERTICAL),
            'neuron_id_y_offset': kwargs.get('neuron_id_y_offset', DEFAULT_NEURON_ID_Y_OFFSET),
            'neuron_id_fontsize': kwargs.get('neuron_id_fontsize', DEFAULT_NEURON_ID_FONTSIZE),

            'linewidth_base': kwargs.get('linewidth_base', DEFAULT_LINEWIDTH_BASE),
            'linewidth_scale_tier1': kwargs.get('linewidth_scale_tier1', DEFAULT_LINEWIDTH_SCALE_TIER1),
            'linewidth_scale_tier2': kwargs.get('linewidth_scale_tier2', DEFAULT_LINEWIDTH_SCALE_TIER2),
            'weight_threshold_tier1': kwargs.get('weight_threshold_tier1', DEFAULT_WEIGHT_THRESHOLD_TIER1),
            'weight_threshold_tier2': kwargs.get('weight_threshold_tier2', DEFAULT_WEIGHT_THRESHOLD_TIER2),
            'max_linewidth': kwargs.get('max_linewidth', DEFAULT_MAX_LINEWIDTH),

            'weight_text_show_threshold': kwargs.get('weight_text_show_threshold', DEFAULT_WEIGHT_TEXT_SHOW_THRESHOLD),
            'weight_text_fontsize': kwargs.get('weight_text_fontsize', DEFAULT_WEIGHT_TEXT_FONTSIZE),
            'weight_text_precision': kwargs.get('weight_text_precision', DEFAULT_WEIGHT_TEXT_PRECISION),
            'weight_text_bbox_alpha': kwargs.get('weight_text_bbox_alpha', DEFAULT_WEIGHT_TEXT_BBOX_ALPHA),
            'weight_text_bbox_width': kwargs.get('weight_text_bbox_width', DEFAULT_WEIGHT_TEXT_BBOX_WIDTH),
            'weight_text_bbox_height': kwargs.get('weight_text_bbox_height', DEFAULT_WEIGHT_TEXT_BBOX_HEIGHT),
            
            'text_overlap_grid_size': kwargs.get('text_overlap_grid_size', DEFAULT_TEXT_OVERLAP_GRID_SIZE),
            'text_label_search_segments': kwargs.get('text_label_search_segments', DEFAULT_TEXT_LABEL_SEARCH_SEGMENTS),
            'text_label_search_start_offset_divisor': kwargs.get('text_label_search_start_offset_divisor', DEFAULT_TEXT_LABEL_SEARCH_START_OFFSET_DIVISOR),

            'layer_label_fontsize': kwargs.get('layer_label_fontsize', DEFAULT_LAYER_LABEL_FONTSIZE),
            'figsize': kwargs.get('figsize', DEFAULT_FIGSIZE),
            'output_filename_prefix': kwargs.get('output_filename_prefix', DEFAULT_OUTPUT_FILENAME_PREFIX),
            'dpi': kwargs.get('dpi', DEFAULT_DPI),
            'title_fontsize': kwargs.get('title_fontsize', DEFAULT_TITLE_FONTSIZE),
            'title': kwargs.get('title', 'Neural Network Architecture') # Custom title for plot
        }

        if self.weights_list is None:
            print("No weights_list provided. Generating default weights for visualization.")
            self._generate_default_weights()
        
        # Validate weights_list structure
        if len(self.neural_network_layers) < 2:
            raise ValueError("Neural network must have at least an input and an output layer (2 layers total).")
        if len(self.weights_list) != len(self.neural_network_layers) - 1:
            raise ValueError(f"Expected {len(self.neural_network_layers) - 1} weight matrices, "
                             f"but got {len(self.weights_list)}.")
        for i, W in enumerate(self.weights_list):
            expected_shape = (self.neural_network_layers[i], self.neural_network_layers[i+1])
            if W.shape != expected_shape:
                raise ValueError(f"Weight matrix at index {i} has shape {W.shape}, "
                                 f"but expected {expected_shape}.")

    def _generate_default_weights(self):
        """
        Generates a default list of weights (all 0.4) if none are provided.
        """
        self.weights_list = []
        for n_in, n_out in zip(self.neural_network_layers[:-1], self.neural_network_layers[1:]):
            # Default weight that is visible but not too large, and below tier1 threshold
            default_weight_value = (DEFAULT_WEIGHT_THRESHOLD_TIER1 - 0.1) if DEFAULT_WEIGHT_THRESHOLD_TIER1 > 0.1 else 0.4
            self.weights_list.append(np.full((n_in, n_out), default_weight_value))
        
    def draw(self):
        """
        Constructs the NeuralNetwork object and calls its draw method.
        """
        if not self.neural_network_layers:
            print("Error: neural_network_layers is empty. Cannot draw.")
            return
            
        widest_layer_neuron_count = max(self.neural_network_layers) if self.neural_network_layers else 0
        if widest_layer_neuron_count == 0 and self.neural_network_layers: # Handle case like [0,0,0]
             widest_layer_neuron_count = 1 # Avoid division by zero, ensure minimal space

        network = NeuralNetwork(widest_layer_neuron_count, self.config)
        for num_neurons in self.neural_network_layers:
            if num_neurons < 0:
                raise ValueError("Number of neurons in a layer cannot be negative.")
            # Allow num_neurons == 0, it just won't draw neurons for that layer, which might be intended for some dynamic structures.
            # Or raise error: if num_neurons == 0: raise ValueError("Layer size cannot be zero.")
            network.add_layer(num_neurons)
        
        network.draw(self.weights_list)

def display_youtube(video_id, video_title="Play on YouTube", thumbnail_quality="hqdefault", display_width_px=360):

    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/{thumbnail_quality}.jpg"
    video_watch_url = f"https://www.youtube.com/watch?v={urllib.parse.quote(video_id)}"

    html_output = f"""
    <div style="width: {display_width_px}px; margin: 10px; display: inline-block; text-align: center; vertical-align: top;">
        <a href="{video_watch_url}" target="_blank" title="{video_title}" style="text-decoration: none; color: inherit; display: block;">
            <div style="position: relative; display: inline-block;">
                <img src="{thumbnail_url}" alt="{video_title}" style="width: 100%; height: auto; border: 1px solid #ccc; display: block;">
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 20%;
                    max-width: 68px;
                    aspect-ratio: 68 / 48;
                    background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/100px-YouTube_full-color_icon_%282017%29.svg.png');
                    background-size: contain;
                    background-repeat: no-repeat;
                    opacity: 0.8;
                    pointer-events: none;
                "></div>
            </div>
            <p style="margin-top: 5px; font-family: sans-serif; font-size: 13px; word-wrap: break-word;">{video_title}</p>
        </a>
    </div>
    """
    display(HTML(html_output))
    # print(f"Thumbnail for video ID '{video_id}' displayed. Click to play on YouTube.") # Optional print

