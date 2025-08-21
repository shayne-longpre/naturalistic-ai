import json
import random
import sys
import argparse
import logging
from collections import defaultdict
from datetime import datetime
from dateutil.parser import parse as parse_date
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

sys.path.append("./")

from src.classes.dataset import Dataset
from src.classes.conversation import Conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwoStageSampler:
    """
    Implements a two-stage sampling approach for conversation data.
    
    Stage 1: Filter to users with conversation counts within bounds
    Stage 2: Temporal stratified sampling across all (filtered) conversations
    Stage 3: User-centric temporal stratified sampling within selected users
    """
    
    def __init__(
        self,
        min_conversations: int = 10,
        max_conversations: int = 200,
        conversations_per_user: int = 30,
        user_temporal_sampling: bool = True,
        seed: int = 42
    ):
        self.min_conversations = min_conversations
        self.max_conversations = max_conversations
        self.conversations_per_user = conversations_per_user
        self.user_temporal_sampling = user_temporal_sampling
        self.seed = seed
        random.seed(seed)
        
    def filter_users_by_conversation_count(
        self, 
        conversations: List[Conversation]
    ) -> Tuple[List[Conversation], Dict[str, int]]:
        """
        Filter conversations to only include those from users with 
        conversation counts within specified bounds.
        
        Returns:
            Tuple of (filtered conversations, user conversation counts)
        """
        # Check if user information exists
        if not conversations or not hasattr(conversations[0], 'user_id'):
            logger.warning("No user information found in conversations")
            return conversations, {}
        
        # Count conversations per user
        user_counts = defaultdict(int)
        for conv in conversations:
            if hasattr(conv, 'user_id') and conv.user_id:
                user_counts[conv.user_id] += 1
        
        # Filter users within bounds
        valid_users = {
            user_id for user_id, count in user_counts.items()
            if self.min_conversations <= count <= self.max_conversations
        }
        
        logger.info(f"Total users: {len(user_counts)}")
        logger.info(f"Users within bounds ({self.min_conversations}-{self.max_conversations} conversations): {len(valid_users)}")
        
        # Filter conversations
        filtered_conversations = [
            conv for conv in conversations
            if hasattr(conv, 'user_id') and conv.user_id in valid_users
        ]
        
        logger.info(f"Conversations after user filtering: {len(filtered_conversations)} (from {len(conversations)})")
        
        return filtered_conversations, dict(user_counts)
    
    def get_month_bin(self, dt: datetime) -> str:
        """Get month bin identifier from datetime."""
        return dt.strftime('%Y-%m')
    
    def get_quarter_bin(self, dt: datetime) -> str:
        """Get quarter bin identifier from datetime."""
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{quarter}"
    
    def temporal_stratified_sample(
        self,
        conversations: List[Conversation],
        sample_size: int
    ) -> Tuple[List[Conversation], Dict[str, float]]:
        """
        Perform stratified random sampling by month.
        
        Returns:
            Tuple of (sampled conversations, sampling weights by month)
        """
        # Group conversations by month
        month_bins = defaultdict(list)
        no_time_convs = []
        
        for conv in conversations:
            try:
                if hasattr(conv, 'time') and conv.time:
                    dt = parse_date(conv.time) if isinstance(conv.time, str) else conv.time
                    if isinstance(dt, datetime):
                        bin_key = self.get_month_bin(dt)
                        month_bins[bin_key].append(conv)
                    else:
                        no_time_convs.append(conv)
                else:
                    no_time_convs.append(conv)
            except Exception as e:
                logger.debug(f"Could not parse time for conversation: {e}")
                no_time_convs.append(conv)
        
        if not month_bins:
            logger.warning("No conversations with valid timestamps found. Falling back to random sampling.")
            return self._simple_random_sample(conversations, sample_size), {}
        
        logger.info(f"Found {len(month_bins)} months with conversations")
        logger.info(f"Conversations without valid timestamps: {len(no_time_convs)}")
        
        # Calculate samples per month
        num_bins = len(month_bins)
        per_bin = sample_size // num_bins
        remainder = sample_size % num_bins
        
        # Sample from each month
        sampled = []
        weights = {}
        
        sorted_months = sorted(month_bins.keys())
        for i, month in enumerate(sorted_months):
            bin_data = month_bins[month]
            n_to_sample = per_bin + (1 if i < remainder else 0)
            n_sampled = min(n_to_sample, len(bin_data))
            
            sampled_from_bin = random.sample(bin_data, n_sampled)
            sampled.extend(sampled_from_bin)
            
            # Calculate weight for this month
            if n_sampled > 0:
                weights[month] = len(bin_data) / n_sampled
            
            logger.info(f"Month {month}: sampled {n_sampled} from {len(bin_data)} conversations (weight: {weights.get(month, 0):.2f})")
        
        return sampled, weights
    
    def _simple_random_sample(
        self,
        conversations: List[Conversation],
        sample_size: int
    ) -> List[Conversation]:
        """Fallback simple random sampling."""
        n_to_sample = min(sample_size, len(conversations))
        return random.sample(conversations, n_to_sample)
    
    def user_centric_sample(
        self,
        conversations: List[Conversation],
        n_users: int,
        prioritize_users: Optional[Set[str]] = None
    ) -> Tuple[List[Conversation], Dict[str, Dict]]:
        """
        Sample users and optionally perform temporal stratified sampling within each user's conversations.
        
        Args:
            conversations: List of conversations to sample from
            n_users: Number of users to sample
            prioritize_users: Set of user IDs to prioritize (from temporal sample)
            
        Returns:
            Tuple of (sampled conversations, user temporal weights by user_id)
        """
        # Check if user information exists
        if not conversations or not hasattr(conversations[0], 'user_id'):
            logger.warning("No user information found. Skipping user-centric sampling.")
            return [], {}
        
        # Group conversations by user
        user_conversations = defaultdict(list)
        for conv in conversations:
            if hasattr(conv, 'user_id') and conv.user_id:
                user_conversations[conv.user_id].append(conv)
        
        all_users = set(user_conversations.keys())
        
        # Prioritize users from temporal sample if provided
        if prioritize_users:
            priority_users = list(prioritize_users & all_users)
            other_users = list(all_users - prioritize_users)
            
            # Sample users
            n_priority = min(len(priority_users), n_users)
            selected_users = random.sample(priority_users, n_priority)
            
            if n_priority < n_users:
                n_other = min(len(other_users), n_users - n_priority)
                selected_users.extend(random.sample(other_users, n_other))
        else:
            selected_users = random.sample(list(all_users), min(n_users, len(all_users)))
        
        logger.info(f"Selected {len(selected_users)} users for user-centric sampling")
        logger.info(f"User temporal sampling: {'enabled (quarterly)' if self.user_temporal_sampling else 'disabled (random)'}")
        
        # Sample from each user's conversations
        sampled = []
        user_temporal_weights = {}
        
        for user_id in selected_users:
            user_convs = user_conversations[user_id]
            n_to_sample = min(self.conversations_per_user, len(user_convs))
            
            # If user has few conversations, take all of them
            if len(user_convs) <= n_to_sample:
                user_sampled = user_convs
                user_temporal_weights[user_id] = {
                    'total_conversations': len(user_convs),
                    'sampled_conversations': len(user_convs),
                    'sampling_method': 'all',
                    'temporal_weights': {}
                }
                logger.debug(f"User {user_id}: sampled all {len(user_convs)} conversations")
            else:
                if self.user_temporal_sampling:
                    # Perform temporal stratified sampling for this user
                    user_sampled, user_weights = self._user_temporal_stratified_sample(
                        user_convs, n_to_sample, user_id
                    )
                    user_temporal_weights[user_id] = {
                        'total_conversations': len(user_convs),
                        'sampled_conversations': len(user_sampled),
                        'sampling_method': 'temporal_stratified_quarterly',
                        'temporal_weights': user_weights
                    }
                else:
                    # Random sampling
                    user_sampled = random.sample(user_convs, n_to_sample)
                    user_temporal_weights[user_id] = {
                        'total_conversations': len(user_convs),
                        'sampled_conversations': len(user_sampled),
                        'sampling_method': 'random',
                        'temporal_weights': {}
                    }
                    logger.debug(f"User {user_id}: randomly sampled {len(user_sampled)} from {len(user_convs)} conversations")
            
            sampled.extend(user_sampled)
            
        logger.info(f"Sampled {len(sampled)} conversations from user-centric sampling")
        
        return sampled, user_temporal_weights
    
    def _user_temporal_stratified_sample(
        self,
        user_conversations: List[Conversation],
        sample_size: int,
        user_id: str
    ) -> Tuple[List[Conversation], Dict[str, float]]:
        """
        Perform temporal stratified sampling within a single user's conversations (by quarter).
        """
        # Group user's conversations by quarter
        quarter_bins = defaultdict(list)
        no_time_convs = []
        
        for conv in user_conversations:
            try:
                if hasattr(conv, 'time') and conv.time:
                    dt = parse_date(conv.time) if isinstance(conv.time, str) else conv.time
                    if isinstance(dt, datetime):
                        bin_key = self.get_quarter_bin(dt)
                        quarter_bins[bin_key].append(conv)
                    else:
                        no_time_convs.append(conv)
                else:
                    no_time_convs.append(conv)
            except Exception:
                no_time_convs.append(conv)
        
        if not quarter_bins:
            # If no valid timestamps, fall back to random sampling
            return random.sample(user_conversations, sample_size), {}
        
        # Calculate samples per quarter for this user
        num_bins = len(quarter_bins)
        per_bin = sample_size // num_bins
        remainder = sample_size % num_bins
        
        # Sample from each quarter
        sampled = []
        weights = {}
        
        sorted_quarters = sorted(quarter_bins.keys())
        for i, quarter in enumerate(sorted_quarters):
            bin_data = quarter_bins[quarter]
            n_to_sample = per_bin + (1 if i < remainder else 0)
            n_sampled = min(n_to_sample, len(bin_data))
            
            if n_sampled > 0:
                sampled_from_bin = random.sample(bin_data, n_sampled)
                sampled.extend(sampled_from_bin)
                weights[quarter] = len(bin_data) / n_sampled
        
        logger.debug(f"User {user_id}: sampled {len(sampled)} conversations across {len(quarter_bins)} quarters")
        
        return sampled, weights
    
    def combine_samples(
        self,
        temporal_sample: List[Conversation],
        user_sample: List[Conversation]
    ) -> Tuple[List[Conversation], Dict[str, Dict]]:
        """
        Combine temporal and user samples, tracking metadata.
        
        Returns:
            Tuple of (combined conversations, metadata about sampling)
        """
        # Track which sample each conversation belongs to
        temporal_ids = {id(conv) for conv in temporal_sample}
        user_ids = {id(conv) for conv in user_sample}
        
        # Combine samples (avoiding duplicates)
        combined = temporal_sample.copy()
        seen_ids = temporal_ids.copy()
        
        for conv in user_sample:
            if id(conv) not in seen_ids:
                combined.append(conv)
                seen_ids.add(id(conv))
        
        # Create metadata for each conversation
        sample_metadata = {}
        for i, conv in enumerate(combined):
            conv_id = id(conv)
            metadata = {
                'index': i,
                'in_temporal_sample': conv_id in temporal_ids,
                'in_user_sample': conv_id in user_ids,
                'in_both_samples': conv_id in temporal_ids and conv_id in user_ids
            }
            
            # Add conversation identifiers if available
            if hasattr(conv, 'conversation_id'):
                metadata['conversation_id'] = conv.conversation_id
            if hasattr(conv, 'user_id'):
                metadata['user_id'] = conv.user_id
            if hasattr(conv, 'time'):
                metadata['time'] = str(conv.time)
            
            sample_metadata[str(i)] = metadata
        
        overlap_count = sum(1 for m in sample_metadata.values() if m['in_both_samples'])
        logger.info(f"Combined sample size: {len(combined)} conversations")
        logger.info(f"Conversations in both samples: {overlap_count}")
        
        return combined, sample_metadata


def main(
    input_path: str,
    output_path: str,
    temporal_sample_size: int,
    n_users: int,
    conversations_per_user: int,
    min_conversations: int,
    max_conversations: int,
    seed: int
):
    """Main sampling pipeline."""
    logger.info("Starting two-stage sampling process")
    logger.info(f"Parameters: temporal_sample_size={temporal_sample_size}, n_users={n_users}, "
                f"conversations_per_user={conversations_per_user}, "
                f"user_bounds=({min_conversations}, {max_conversations})")
    
    # Load dataset
    dataset = Dataset.load(input_path)
    logger.info(f"Loaded dataset with {len(dataset.data)} conversations")
    
    # Initialize sampler
    sampler = TwoStageSampler(
        min_conversations=min_conversations,
        max_conversations=max_conversations,
        conversations_per_user=conversations_per_user,
        seed=seed
    )
    
    # Stage 1: Filter users
    filtered_convs, user_counts = sampler.filter_users_by_conversation_count(dataset.data)
    
    # Stage 2: Temporal stratified sampling
    temporal_sample, temporal_weights = sampler.temporal_stratified_sample(
        filtered_convs, temporal_sample_size
    )
    
    # Get users from temporal sample for prioritization
    temporal_users = set()
    for conv in temporal_sample:
        if hasattr(conv, 'user_id') and conv.user_id:
            temporal_users.add(conv.user_id)
    
    # Stage 3: User-centric sampling with temporal stratification
    user_sample, _ = sampler.user_centric_sample(
        filtered_convs, n_users, prioritize_users=temporal_users
    )
    
    # Combine samples
    combined_sample, sample_metadata = sampler.combine_samples(temporal_sample, user_sample)
    
    # Save sampled dataset
    output_dict = {
        'dataset_id': combined_sample[0].dataset_id,  #dataset['dataset_id'],
        'data': [conv.to_dict() for conv in combined_sample]
    }
    
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    logger.info(f"Saved {len(combined_sample)} sampled conversations to {output_path}")
    
    # Save metadata
    metadata = {
        'sampling_parameters': {
            'temporal_sample_size': temporal_sample_size,
            'n_users': n_users,
            'conversations_per_user': conversations_per_user,
            'min_conversations_per_user': min_conversations,
            'max_conversations_per_user': max_conversations,
            'seed': seed
        },
        'sampling_statistics': {
            'original_dataset_size': len(dataset.data),
            'filtered_dataset_size': len(filtered_convs),
            'temporal_sample_size': len(temporal_sample),
            'user_sample_size': len(user_sample),
            'combined_sample_size': len(combined_sample),
            'overlap_size': sum(1 for m in sample_metadata.values() if m['in_both_samples']),
            'total_users_in_dataset': len(user_counts),
            'users_within_bounds': len([u for u, c in user_counts.items() 
                                       if min_conversations <= c <= max_conversations]),
            'users_in_temporal_sample': len(temporal_users),
            'users_in_user_sample': len(set(conv.user_id for conv in user_sample 
                                           if hasattr(conv, 'user_id') and conv.user_id))
        },
        'temporal_weights': temporal_weights,
        'conversation_metadata': sample_metadata
    }
    
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved sampling metadata to {metadata_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("SAMPLING SUMMARY")
    print("="*50)
    print(f"Original dataset size: {len(dataset.data)}")
    print(f"After user filtering: {len(filtered_convs)}")
    print(f"Temporal sample: {len(temporal_sample)}")
    print(f"User sample: {len(user_sample)}")
    print(f"Combined sample: {len(combined_sample)}")
    print(f"Overlap: {metadata['sampling_statistics']['overlap_size']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage scientific sampling for conversation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input JSON dataset')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save sampled JSON dataset')
    
    # Sampling parameters
    parser.add_argument('--temporal_sample_size', type=int, default=4000,
                       help='Size of temporal stratified sample')
    parser.add_argument('--n_users', type=int, default=100,
                       help='Number of users to sample for user-centric analysis')
    parser.add_argument('--conversations_per_user', type=int, default=30,
                       help='Maximum conversations to sample per user')
    
    # User filtering parameters
    parser.add_argument('--min_conversations', type=int, default=10,
                       help='Minimum conversations per user to include')
    parser.add_argument('--max_conversations', type=int, default=200,
                       help='Maximum conversations per user to include')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        temporal_sample_size=args.temporal_sample_size,
        n_users=args.n_users,
        conversations_per_user=args.conversations_per_user,
        min_conversations=args.min_conversations,
        max_conversations=args.max_conversations,
        seed=args.seed
    )
