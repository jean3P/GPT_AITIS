# models/chunking/factory.py

from typing import Dict, Any, Optional, Type, List
import logging

from .graph_chunker import GraphChunker
from .semantic_graph_chunker import SemanticGraphChunker
from .simple_chunker import SimpleChunker
from .base import ChunkingStrategy, ChunkingError

logger = logging.getLogger(__name__)


class ChunkingFactory:
    """
    Factory for creating chunking strategies.

    Supports easy registration of new strategies and configuration-based creation.
    """

    _strategies: Dict[str, Type[ChunkingStrategy]] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[ChunkingStrategy],
                          default_config: Optional[Dict[str, Any]] = None):
        """
        Register a new chunking strategy.

        Args:
            name: Strategy name (used for selection)
            strategy_class: The strategy class
            default_config: Default configuration for this strategy
        """
        cls._strategies[name] = strategy_class
        cls._default_configs[name] = default_config or {}
        logger.info(f"Registered chunking strategy: {name}")

    @classmethod
    def create_strategy(cls, strategy_name: str,
                        config: Optional[Dict[str, Any]] = None) -> ChunkingStrategy:
        """
        Create a chunking strategy instance.

        Args:
            strategy_name: Name of the strategy to create
            config: Optional configuration to override defaults

        Returns:
            Configured ChunkingStrategy instance

        Raises:
            ChunkingError: If strategy is not found or creation fails
        """
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ChunkingError(
                f"Unknown chunking strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._strategies[strategy_name]
        default_config = cls._default_configs[strategy_name].copy()

        # Merge provided config with defaults
        if config:
            default_config.update(config)

        try:
            strategy = strategy_class(default_config)

            # Validate configuration
            if not strategy.validate_config():
                raise ChunkingError(f"Invalid configuration for strategy: {strategy_name}")

            logger.info(f"Created {strategy_name} chunking strategy")
            return strategy

        except Exception as e:
            raise ChunkingError(f"Failed to create {strategy_name} strategy: {str(e)}")

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy information
        """
        if strategy_name not in cls._strategies:
            raise ChunkingError(f"Unknown strategy: {strategy_name}")

        strategy_class = cls._strategies[strategy_name]
        default_config = cls._default_configs[strategy_name]

        # Create temporary instance to get info
        try:
            temp_instance = strategy_class(default_config)
            strategy_info = temp_instance.get_strategy_info()
            strategy_info['default_config'] = default_config
            return strategy_info
        except Exception as e:
            return {
                'name': strategy_name,
                'class': strategy_class.__name__,
                'error': f"Could not get info: {str(e)}",
                'default_config': default_config
            }

    @classmethod
    def get_all_strategies_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered strategies."""
        return {
            name: cls.get_strategy_info(name)
            for name in cls._strategies.keys()
        }


# Configuration presets for common use cases
CHUNKING_PRESETS = {
    'fast': {
        'strategy': 'simple',
        'config': {'max_length': 256, 'overlap': 25}
    },
    'balanced': {
        'strategy': 'section',
        'config': {'max_section_length': 1500, 'preserve_subsections': True}
    },
    'comprehensive': {
        'strategy': 'section',
        'config': {
            'max_section_length': 2500,
            'preserve_subsections': True,
            'include_front_matter': True,
            'sentence_window_size': 5
        }
    },
    'adaptive': {
        'strategy': 'smart_size',
        'config': {
            'base_chunk_words': 80,
            'min_chunk_words': 25,
            'max_chunk_words': 180,
            'importance_multiplier': 1.4,
            'preserve_complete_clauses': True,
            'overlap_words': 5
        }
    },
    'semantic': {
        'strategy': 'semantic',
        'config': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_value': 75,
            'min_chunk_sentences': 2,
            'max_chunk_sentences': 15,
            'preserve_paragraph_boundaries': True
        }
    },
    'semantic_focused': {
        'strategy': 'semantic',
        'config': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_value': 85,  # Higher threshold = fewer breaks
            'min_chunk_sentences': 3,
            'max_chunk_sentences': 12,
            'preserve_paragraph_boundaries': True
        }
    },
    'semantic_comprehensive': {
        'strategy': 'semantic',
        'config': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_value': 65,  # Lower threshold = more breaks
            'min_chunk_sentences': 2,
            'max_chunk_sentences': 20,
            'preserve_paragraph_boundaries': True
        }
    },
    'insurance_semantic': {
        'strategy': 'semantic',
        'config': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_value': 70,
            'min_chunk_sentences': 2,
            'max_chunk_sentences': 15,
            'preserve_paragraph_boundaries': True,
            'device': 'cpu'
        }
    },
    'graph': {
        'strategy': 'graph',
        'config': {
            'max_chunk_size': 512,
            'community_size': 50,
            'enable_hierarchical': True
        }
    },
    'graph_simple': {
        'strategy': 'graph',
        'config': {
            'max_chunk_size': 256,
            'community_size': 30,
            'enable_hierarchical': False
        }
    },
    'graph_comprehensive': {
        'strategy': 'graph',
        'config': {
            'max_chunk_size': 1024,
            'community_size': 100,
            'enable_hierarchical': True
        }
    }
}


# Missing functions that vector_store.py is trying to import
def create_preset_strategy(preset_name: str,
                           config_overrides: Optional[Dict[str, Any]] = None) -> ChunkingStrategy:
    """
    Create a chunking strategy from a preset configuration.

    Args:
        preset_name: Name of the preset to use
        config_overrides: Optional configuration overrides

    Returns:
        Configured ChunkingStrategy instance

    Raises:
        ChunkingError: If preset is not found or strategy creation fails
    """
    if preset_name not in CHUNKING_PRESETS:
        available_presets = list(CHUNKING_PRESETS.keys())
        raise ChunkingError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {available_presets}"
        )

    preset_config = CHUNKING_PRESETS[preset_name]
    strategy_name = preset_config['strategy']
    strategy_config = preset_config['config'].copy()

    # Apply config overrides if provided
    if config_overrides:
        strategy_config.update(config_overrides)

    # Create the strategy using the factory
    return ChunkingFactory.create_strategy(strategy_name, strategy_config)


def get_available_presets() -> List[str]:
    """Get list of available preset names."""
    return list(CHUNKING_PRESETS.keys())


def get_preset_info(preset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific preset.

    Args:
        preset_name: Name of the preset

    Returns:
        Dictionary with preset information
    """
    if preset_name not in CHUNKING_PRESETS:
        raise ChunkingError(f"Unknown preset: {preset_name}")

    preset_config = CHUNKING_PRESETS[preset_name]
    return {
        'name': preset_name,
        'strategy': preset_config['strategy'],
        'config': preset_config['config'],
        'description': f"Preset configuration for {preset_config['strategy']} strategy"
    }


def get_all_presets_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all available presets."""
    return {
        name: get_preset_info(name)
        for name in CHUNKING_PRESETS.keys()
    }


# Auto-registration system
def auto_register_strategies():
    """
    Automatically register all available chunking strategies.
    This function is called when the module is imported.
    """
    try:
        # Register simple chunker (always available)
        ChunkingFactory.register_strategy(
            'simple',
            SimpleChunker,
            {'max_length': 512, 'overlap': 0, 'preserve_paragraphs': True}
        )
        logger.info("Successfully registered SimpleChunker strategy")

        # Register section chunker
        try:
            from .section_chunker import SectionChunker
            ChunkingFactory.register_strategy(
                'section',
                SectionChunker,
                {
                    'max_section_length': 2000,
                    'min_section_length': 50,
                    'preserve_subsections': True,
                    'include_front_matter': False,
                    'sentence_window_size': 3
                }
            )
            logger.info("Successfully registered SectionChunker strategy")
        except ImportError:
            logger.debug("SectionChunker not available")

        # Register smart size chunker
        try:
            from .smart_size_chunker import SmartSizeChunker
            ChunkingFactory.register_strategy(
                'smart_size',
                SmartSizeChunker,
                {
                    'base_chunk_words': 80,
                    'min_chunk_words': 20,
                    'max_chunk_words': 200,
                    'importance_multiplier': 1.5,
                    'coherence_threshold': 0.7,
                    'preserve_complete_clauses': True,
                    'overlap_words': 0
                }
            )
            logger.info("Successfully registered SmartSizeChunker strategy")
        except ImportError:
            logger.debug("SmartSizeChunker not available")

        # Register semantic chunker
        try:
            from .semantic_chunker import SemanticChunker
            ChunkingFactory.register_strategy(
                'semantic',
                SemanticChunker,
                {
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'breakpoint_threshold_type': 'percentile',
                    'breakpoint_threshold_value': 75,
                    'min_chunk_sentences': 2,
                    'max_chunk_sentences': 15,
                    'buffer_size': 1,
                    'preserve_paragraph_boundaries': True,
                    'device': 'cpu'
                }
            )
            logger.info("Successfully registered SemanticChunker strategy")
        except ImportError:
            logger.debug("SemanticChunker not available")

        # NEW: Register graph chunker for PankRAG
        try:
            ChunkingFactory.register_strategy(
                'graph',
                GraphChunker,
                {
                    'max_chunk_size': 512,
                    'community_size': 50,
                    'enable_hierarchical': True
                }
            )
            logger.info("Successfully registered GraphChunker strategy")
        except ImportError:
            logger.debug("GraphChunker not available")

        try:
            ChunkingFactory.register_strategy(
                'semantic_graph',
                SemanticGraphChunker,
                {
                    'max_chunk_size': 512,
                    'similarity_threshold': 0.7,
                    'min_community_size': 3,
                    'enable_hierarchical': True,
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'semantic_window': 5,
                    'entity_weight': 1.5,
                    'sequential_weight': 0.8
                }
            )
            logger.info("Successfully registered SemanticGraphChunker strategy")
        except ImportError as e:
            logger.debug(f"SemanticGraphChunker not available: {e}")

    except Exception as e:
        logger.error(f"Error during auto-registration: {e}")


# Auto-register strategies when module is imported
auto_register_strategies()
