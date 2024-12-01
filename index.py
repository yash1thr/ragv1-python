"""Runs a batch job to compute embeddings for an entire repo and stores them into a vector store."""

import logging
import os
import time
import config as sage_config

import configargparse
from data_manager import GitHubRepoManager
from chunker import helper
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = configargparse.ArgParser(
        description="Batch-embeds a GitHub repository and its issues.", ignore_unknown_config_file_keys=True
    )
    sage_config.add_config_args(parser)

    arg_validators = [
        sage_config.add_repo_args(parser),
        sage_config.add_indexing_args(parser),
    ]

    args = parser.parse_args()

    for validator in arg_validators:
        validator(args)

    

    ######################
    # Step 1: Embeddings #
    ######################

    # Index the repository.
    repo_embedder = None
    if args.index_repo:
        logging.info("Cloning the repository...")
        repo_manager = GitHubRepoManager.from_args(args)
        helper(repo_manager)
        logging.info(repo_manager.local_dir)
        logging.info("Embedding the repo...")
        #chunker = UniversalFileChunker(max_tokens=args.tokens_per_chunk)
        #repo_embedder = build_batch_embedder_from_flags(repo_manager, chunker, args)
        #repo_jobs_file = repo_embedder.embed_dataset(args.chunks_per_batch, args.max_embedding_jobs)

    logging.info("Done!")


if __name__ == "__main__":
    main()
