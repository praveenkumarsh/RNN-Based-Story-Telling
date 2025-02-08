// Define the VPC
resource "aws_vpc" "main" {
  cidr_block = var.cidr_block

  tags = {
    Name = var.vpc_name
  }
}

// Define public subnets
resource "aws_subnet" "public_subnet" {
  count             = length(var.azs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.cidr_block, 8, count.index)
  availability_zone = element(var.azs, count.index)
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.vpc_name}-subnet-public${count.index + 1}-${element(var.azs, count.index)}"
  }
}

// Define private subnets
resource "aws_subnet" "private_subnet" {
  count             = length(var.azs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.cidr_block, 8, count.index + length(var.azs))
  availability_zone = element(var.azs, count.index)

  tags = {
    Name = "${var.vpc_name}-subnet-private${count.index + 1}-${element(var.azs, count.index)}"
  }
}

// Define the internet gateway
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.vpc_name}-igw"
  }
}

// Define the NAT gateway
resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = element(aws_subnet.public_subnet[*].id, 0)

  tags = {
    Name = "${var.vpc_name}-nat"
  }
}

// Define the Elastic IP for the NAT gateway
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name = "${var.vpc_name}-eip"
  }
}

// Define the public route table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "${var.vpc_name}-rtb-public"
  }
}

// Associate public subnets with the public route table
resource "aws_route_table_association" "public" {
  count          = length(var.azs)
  subnet_id      = element(aws_subnet.public_subnet[*].id, count.index)
  route_table_id = aws_route_table.public.id
}

// Define the private route table for each AZ
resource "aws_route_table" "private" {
  count  = length(var.azs)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }

  tags = {
    Name = "${var.vpc_name}-rtb-private${count.index + 1}-${element(var.azs, count.index)}"
  }
}

// Associate private subnets with the private route table
resource "aws_route_table_association" "private" {
  count          = length(var.azs)
  subnet_id      = element(aws_subnet.private_subnet[*].id, count.index)
  route_table_id = element(aws_route_table.private[*].id, count.index)
}

// Define the S3 VPC endpoint
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.region}.s3"
  route_table_ids = concat([aws_route_table.public.id], aws_route_table.private[*].id)

  tags = {
    Name = "${var.vpc_name}-vpce-s3"
  }
}