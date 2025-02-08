variable "cidr_block" {
  description = "The CIDR block for the VPC"
  type        = string
}

variable "azs" {
  description = "The availability zones for the VPC"
  type        = list(string)
}

variable "vpc_name" {
  description = "The name of the VPC"
  type        = string
}

variable "region" {
  description = "The AWS region"
  type        = string
}