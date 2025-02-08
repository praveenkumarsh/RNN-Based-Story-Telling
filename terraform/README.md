# Terraform ECS Project

This project sets up an Amazon ECS (Elastic Container Service) environment using Terraform. It includes the creation of a VPC, security group, ECS cluster, and ECS service, along with a GitHub Actions workflow for deployment.

## Project Structure

```
terraform-ecs-project
├── modules
│   ├── ecs-cluster        # Module for ECS cluster
│   ├── ecs-service        # Module for ECS service
│   ├── vpc                # Module for VPC
│   └── security-group     # Module for security group
├── .github
│   └── workflows
│       └── deploy.yml     # GitHub Actions workflow for deployment
├── main.tf                # Entry point for Terraform project
├── outputs.tf             # Outputs for the entire project
├── variables.tf           # Input variables for the project
└── README.md              # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**: Clone this repository to your local machine.
   
   ```bash
   git clone <repository-url>
   cd terraform-ecs-project
   ```

2. **Configure Variables**: Update the `variables.tf` file with your desired configuration, including region and environment settings.

3. **Initialize Terraform**: Run the following command to initialize Terraform and download the necessary providers.

   ```bash
   terraform init
   ```

4. **Plan the Deployment**: Generate an execution plan to see what resources will be created.

   ```bash
   terraform plan
   ```

5. **Apply the Configuration**: Apply the Terraform configuration to create the resources.

   ```bash
   terraform apply
   ```

## GitHub Actions Workflow

The project includes a GitHub Actions workflow located in `.github/workflows/deploy.yml`. This workflow automates the deployment process whenever changes are pushed to the main branch.

## Usage

After the resources are created, you can deploy your application to the ECS service. Refer to the individual module README files for more detailed usage instructions.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.