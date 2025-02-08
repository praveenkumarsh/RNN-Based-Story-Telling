provider "aws" {
  region = var.region
}

resource "aws_ecs_cluster" "this" {
  name = var.cluster_name
}