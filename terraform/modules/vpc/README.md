# VPC Module

This module is responsible for creating a Virtual Private Cloud (VPC) in AWS. It sets up the necessary networking components, including subnets and route tables, to facilitate the deployment of resources within the VPC.

## Usage

To use this module, include it in your Terraform configuration as follows:

```hcl
module "vpc" {
  source          = "./modules/vpc"
  cidr_block      = var.cidr_block
  availability_zones = var.availability_zones
}
```

## Inputs

| Name                | Description                          | Type          | Default | Required |
|---------------------|--------------------------------------|---------------|---------|:--------:|
| cidr_block          | The CIDR block for the VPC          | string        | n/a     | yes      |
| availability_zones  | List of availability zones           | list(string)  | n/a     | yes      |

## Outputs

| Name                | Description                          |
|---------------------|--------------------------------------|
| vpc_id              | The ID of the created VPC           |
| subnet_ids          | List of subnet IDs created in the VPC|