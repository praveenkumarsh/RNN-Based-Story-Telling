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

module "ecs_cluster" {
  source       = "./modules/ecs-cluster"
  region       = var.region
  cluster_name = var.cluster_name
}

module "ecs_task_definition" {
  source = "./modules/ecs-task-definition"
  region       = var.region
}

module "ecs_service" {
  source            = "./modules/ecs-service"
  region            = var.region
  service_name      = var.service_name
  cluster_id        = module.ecs_cluster.cluster_id
  desired_count     = var.desired_count
  container_name    = var.container_name
  container_port    = var.container_port
  task_definition   = module.ecs_task_definition.ecs_task_definition_arn

}
