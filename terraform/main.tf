provider "aws" {
  region = var.region
}

module "vpc" {
  source     = "./modules/vpc"
  cidr_block = var.vpc_cidr
  azs        = var.availability_zones
  vpc_name   = var.vpc_name
  region     = var.region
}

module "security_group" {
  source     = "./modules/security-group"
  allowed_ip = var.allowed_ip
  vpc_id     = module.vpc.vpc_id
}

resource "aws_ecs_cluster" "main" {
  name = var.cluster_name
}

output "cluster_arn" {
  value = aws_ecs_cluster.main.arn
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}
