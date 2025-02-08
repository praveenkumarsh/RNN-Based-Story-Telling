variable "region" {
  description = "The AWS region to deploy the ECS service"
  type        = string
}

variable "service_name" {
  description = "The name of the ECS service"
  type        = string
}

variable "cluster_id" {
  description = "The ID of the ECS cluster"
  type        = string
}

variable "task_definition" {
  description = "The task definition to use for the ECS service"
  type        = string
}

variable "desired_count" {
  description = "The desired number of tasks"
  type        = number
}

variable "container_name" {
  description = "The name of the container"
  type        = string
}

variable "container_port" {
  description = "The port on which the container is listening"
  type        = number
}

variable "ecs_task_definition_arn" {
  type        = string
  default = "value"
}