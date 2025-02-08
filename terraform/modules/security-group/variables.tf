variable "allowed_ip" {
  description = "The IP address allowed to access the security group"
  type        = string
}

variable "vpc_id" {
  description = "The ID of the VPC"
  type        = string
}