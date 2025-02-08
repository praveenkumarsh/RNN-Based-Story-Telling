variable "region" {
  description = "The AWS region to deploy resources"
  type        = string
}

variable "vpc_cidr" {
  description = "The CIDR block for the VPC"
  type        = string
}

variable "availability_zones" {
  description = "The availability zones for the VPC"
  type        = list(string)
}

variable "allowed_ip" {
  description = "The IP address allowed to access the security group"
  type        = string
}

variable "cluster_name" {
  description = "The name of the ECS cluster"
  type        = string
}

variable "task_family" {
  description = "The family of the ECS task definition"
  type        = string
}

variable "cpu" {
  description = "The number of CPU units used by the task"
  type        = string
}

variable "memory" {
  description = "The amount of memory (in MiB) used by the task"
  type        = string
}

variable "container_name" {
  description = "The name of the container"
  type        = string
}

variable "image" {
  description = "The Docker image to use for the container"
  type        = string
}

variable "container_port" {
  description = "The port on which the container will listen"
  type        = number
}

variable "service_name" {
  description = "The name of the ECS service"
  type        = string
}

variable "desired_count" {
  description = "The number of tasks to run in the ECS service"
  type        = number
}

variable "vpc_name" {
  description = "The name of the VPC"
  type        = string
}