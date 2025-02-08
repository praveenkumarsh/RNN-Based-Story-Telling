# Security Group Module

This module is responsible for creating a security group in AWS. It defines the necessary rules to allow all outbound traffic and restricts inbound traffic to a specified IP address.

## Usage

To use this module, include it in your Terraform configuration as follows:

```hcl
module "security_group" {
  source = "./modules/security-group"

  allowed_ip = "YOUR_IP_ADDRESS"
}
```

## Inputs

| Name        | Description                          | Type   | Default | Required |
|-------------|--------------------------------------|--------|---------|:--------:|
| allowed_ip  | The IP address that is allowed inbound access. | string | n/a     | yes      |

## Outputs

| Name                | Description                          |
|---------------------|--------------------------------------|
| security_group_id   | The ID of the created security group. |