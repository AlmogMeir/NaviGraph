"""Clean session class for NaviGraph unified plugin architecture."""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
from loguru import logger
import pickle

# Type alias for logger
Logger = type(logger)

from .exceptions import NavigraphError
from .navigraph_plugin import NaviGraphPlugin
from .session_visualizer import SessionVisualizer


class Session:
    """Core session that orchestrates unified plugin execution."""
    
    def __init__(
        self,
        session_configuration: Dict[str, Any],
        logger_instance: Logger
    ):
        """Initialize session with plugin orchestration.
        
        Args:
            session_configuration: Configuration dict with:
                - session_id: Session identifier
                - experiment_path: Path to experiment root
                - plugins: List of plugin specifications
                - graph: Graph configuration (with mandatory mapping_file)
            logger_instance: Logger for this session
        """
        self.session_id = session_configuration.get('session_id', 'unknown_session')
        self.config = session_configuration
        self.experiment_path = Path(session_configuration.get('experiment_path', '.'))
        self.session_path = self.experiment_path / self.session_id
        self.logger = logger_instance
        
        # Plugin specifications from config
        self.plugin_specifications = session_configuration.get('plugins', [])
        
        # Shared resources populated by graph and plugins
        self.shared_resources = {}
        
        # Plugin instances stored by name (allows multiple instances of same type)
        self.plugin_instances: Dict[str, NaviGraphPlugin] = {}
        
        # Cached integrated dataframe
        self._integrated_dataframe: Optional[pd.DataFrame] = None
        
        # Initialize orchestrators (auto-discovery happens when registry module is imported)
        self.visualizer = SessionVisualizer(session_configuration)

        # Initialize session
        try:
            self._create_graph_structure()
            self._load_plugins()
            self.logger.info(f"Session {self.session_id} initialized: {len(self.plugin_instances)} plugins")
        except Exception as e:
            raise NavigraphError(f"Session initialization failed: {str(e)}") from e
    
    def get_integrated_dataframe(self) -> pd.DataFrame:
        """Get integrated DataFrame with all plugin augmentations.
        
        Returns:
            DataFrame with all plugin-added columns
        """
        if self._integrated_dataframe is None:
            self._integrated_dataframe = self._execute_plugins()
        return self._integrated_dataframe
    
    def get_shared_resources(self) -> Dict[str, Any]:
        """Get shared resources dictionary.
        
        Returns:
            Copy of shared resources
        """
        return self.shared_resources.copy()
    
    def get_session_id(self) -> str:
        """Get session identifier.
        
        Returns:
            Session ID string
        """
        return self.session_id
    
    def _create_graph_structure(self) -> None:
        """Create graph structure from config and load mandatory spatial mapping."""
        graph_config = self.config.get('graph')
        if not graph_config:
            raise NavigraphError(
                "Graph configuration is required. Please provide 'graph' section in config with "
                "'builder' and 'mapping_file' specifications."
            )
        
        try:
            # Import graph components
            from .graph.structures import GraphStructure
            from .graph.mapping import SpatialMapping
            
            # Get builder configuration
            builder = graph_config.get('builder', {})
            builder_type = builder.get('type')
            builder_params = builder.get('config', {})
            
            if not builder_type:
                raise NavigraphError("Graph builder type is required in graph.builder.type")
            
            # Create graph structure using from_config
            graph = GraphStructure.from_config(builder_type, builder_params)
            
            # Get mapping file - this is MANDATORY
            mapping_file = graph_config.get('mapping_file')
            if not mapping_file:
                raise NavigraphError(
                    "Graph mapping file is required. Please provide 'mapping_file' in graph config. "
                    "Use 'navigraph setup graph' command to create a mapping file."
                )
            
            # Create spatial mapping with conflict strategy
            conflict_strategy = graph_config.get('conflict_strategy', 'node_priority')
            mapping = SpatialMapping(graph, conflict_strategy=conflict_strategy)
            
            # Load mapping from file (mandatory)
            mapping_path = self.experiment_path / mapping_file
            if not mapping_path.exists():
                raise NavigraphError(
                    f"Mapping file not found: {mapping_path}\n"
                    f"Please ensure the mapping file exists or create one using 'navigraph setup graph' command."
                )
            
            # Load the mapping data
            with open(mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
            
            # Check if mapping file has builder info and validate it matches config
            if 'builder' in mapping_data:
                file_builder = mapping_data['builder']
                file_builder_type = file_builder.get('type')
                file_builder_config = file_builder.get('config', {})
                
                # Check if builder type matches
                if file_builder_type != builder_type:
                    self.logger.error(
                        f"⚠️ BUILDER MISMATCH: Mapping file was created with '{file_builder_type}' builder "
                        f"but config specifies '{builder_type}' builder. This may cause incorrect graph mappings!"
                    )
                    raise NavigraphError(
                        f"Builder type mismatch: mapping file uses '{file_builder_type}' "
                        f"but config uses '{builder_type}'. Please regenerate mapping or update config."
                    )
                
                # Warn if builder config differs
                if file_builder_config != builder_params:
                    self.logger.warning(
                        f"⚠️ BUILDER CONFIG DIFFERS: Mapping file builder config differs from current config.\n"
                        f"  Mapping file: {file_builder_config}\n"
                        f"  Current config: {builder_params}\n"
                        f"This may cause incorrect spatial mappings!"
                    )
                    # Don't fail, just warn - config differences might be intentional
            else:
                self.logger.warning(
                    f"⚠️ Mapping file has no builder information - cannot validate compatibility. "
                    f"Consider regenerating the mapping file."
                )
            
            # Load mapping using from_simple_format
            if 'mappings' not in mapping_data:
                raise NavigraphError(
                    f"Invalid mapping file format: {mapping_file}. "
                    f"Missing 'mappings' key. Please recreate using 'navigraph setup graph'."
                )
            
            mapping.from_simple_format(mapping_data['mappings'])
            
            # Validate mapping has actual mappings
            mapped_nodes = len(mapping.get_mapped_nodes())
            mapped_edges = len(mapping.get_mapped_edges())
            
            if mapped_nodes == 0 and mapped_edges == 0:
                raise NavigraphError(
                    f"Mapping file {mapping_file} contains no spatial mappings. "
                    f"Please create mappings using 'navigraph setup graph' command."
                )
            
            self.logger.info(
                f"Loaded spatial mapping: {mapped_nodes} nodes, {mapped_edges} edges mapped"
            )
            
            # Add to shared resources (first resources)
            self.shared_resources['graph'] = graph
            self.shared_resources['graph_mapping'] = mapping
            self.shared_resources['graph_metadata'] = graph_config.get('metadata', {})
            
            self.logger.info(f"Graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")
            
        except Exception as e:
            self.logger.error(f"Graph creation failed: {str(e)}")
            raise
    
    def _load_plugins(self) -> None:
        """Load plugin instances from specifications."""
        if not self.plugin_specifications:
            self.logger.warning("No plugins specified")
            return
        
        # Import plugin registry
        from .registry import registry
        
        for spec in self.plugin_specifications:
            plugin_name = spec.get('name', 'unnamed')
            plugin_type = spec.get('type')
            
            # Skip disabled plugins
            if not spec.get('enable', True):
                self.logger.info(f"Skipping disabled: {plugin_name}")
                continue
            
            try:
                # Get plugin class from registry
                plugin_class = registry.get_plugin_class(plugin_type)
                if not plugin_class:
                    raise NavigraphError(f"Unknown plugin type: {plugin_type}")
                
                # Create plugin instance
                plugin = plugin_class(spec, self.session_path, self.experiment_path)
                
                # Store by name (allows multiple instances of same type)
                if plugin_name in self.plugin_instances:
                    self.logger.warning(f"Overwriting plugin with duplicate name: {plugin_name}")
                
                self.plugin_instances[plugin_name] = plugin
                self.logger.info(f"Loaded: {plugin_name} ({plugin_type})")
                
            except Exception as e:
                error = f"Failed to load {plugin_name}: {str(e)}"
                if spec.get('required', True):
                    raise NavigraphError(error) from e
                else:
                    self.logger.warning(f"Optional plugin failed: {error}")
    
    def _execute_plugins(self) -> pd.DataFrame:
        """Execute plugins in two phases: provide then augment_data.
        
        Returns:
            Integrated DataFrame
        """
        if not self.plugin_instances:
            self.logger.warning("No plugins - returning empty DataFrame")
            return pd.DataFrame()
        
        self.logger.info(f"Executing {len(self.plugin_instances)} plugins")
        
        # Phase 1: Provide
        self.logger.info("PHASE 1: Provide")
        for name, plugin in self.plugin_instances.items():
            try:
                plugin.provide(self.shared_resources)
                self.logger.debug(f"✓ {name} provide complete")
            except Exception as e:
                error = f"{name} provide failed: {str(e)}"
                if plugin.config.get('required', True):
                    raise NavigraphError(error) from e
                self.logger.warning(error)
        
        self.logger.info(f"Resources available: {list(self.shared_resources.keys())}")
        
        # Phase 2: Augment Data
        self.logger.info("PHASE 2: Augment Data")
        dataframe = pd.DataFrame()
        
        for name, plugin in self.plugin_instances.items():
            try:
                cols_before = len(dataframe.columns)
                dataframe = plugin.augment_data(dataframe, self.shared_resources)
                cols_added = len(dataframe.columns) - cols_before
                
                if cols_added > 0:
                    self.logger.info(f"✓ {name}: +{cols_added} columns")
                else:
                    self.logger.debug(f"✓ {name}: no columns added")
                    
            except Exception as e:
                error = f"{name} augment failed: {str(e)}"
                if plugin.config.get('required', True):
                    raise NavigraphError(error) from e
                self.logger.warning(error)
        
        self.logger.info(f"Complete: {len(dataframe.columns)} columns, {len(dataframe)} frames")
        return dataframe
    
    
    def create_visualization(self, video_path: Optional[str] = None, output_name: Optional[str] = None, output_dir: Optional[str] = None, show_realtime: bool = False) -> Optional[str]:
        """Create visualization through SessionVisualizer.
        
        Args:
            video_path: Path to input video (auto-detected if None)
            output_name: Name for output file (defaults to session_id)
            output_dir: Output directory path (uses config if None)
            show_realtime: Whether to display frames in real-time during processing
            
        Returns:
            Path to created video or None
        """
        if self._integrated_dataframe is None:
            self._integrated_dataframe = self._execute_plugins()
        
        # Auto-detect video if not provided
        if video_path is None:
            video_path = self.visualizer.find_video(self.session_path)
            if video_path is None:
                self.logger.error("No video file found in session directory")
                return None
            video_path = str(video_path)
        
        output_name = output_name or self.session_id
        
        return self.visualizer.process_video(
            video_path=video_path,
            dataframe=self._integrated_dataframe,
            shared_resources=self.shared_resources,
            output_name=output_name,
            output_dir=output_dir,
            show_realtime=show_realtime
        )